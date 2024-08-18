import math
import numpy as np
import torch
from torch import nn
from torch.optim import _functional as F
import json
import copy
from PIL import Image
from random import randint
from tqdm import tqdm
import wandb

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import o3d_knn, setup_camera

from torch.optim.lr_scheduler import LambdaLR

from helpers import quat_mult, weighted_l2_loss_v2, l1_loss_v2
from external import calc_ssim, build_rotation

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import sys

# T = 100

def get_dataset(t, md, seq):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"./data/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        seg = np.array(copy.deepcopy(Image.open(f"./data/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})
    return dataset


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data

def params2rendervar(params):
    rendervar = {
        'means3D': params['means'],
        'colors_precomp': params['colors'],
        'rotations': torch.nn.functional.normalize(params['rotations']),
        'opacities': torch.sigmoid(params['opacities']),
        'scales': torch.exp(params['scales']),
        'means2D': torch.zeros_like(params['means'], requires_grad=True, device="cuda") + 0
    }
    return rendervar

def get_loss(params, batch, variables, alpha):
    rendervar = params2rendervar(params)
    # rendervar['means2D'].retain_grad()

    im, _, _, = Renderer(raster_settings=batch['cam'])(**rendervar)

    is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
    fg_pts = rendervar['means3D'][is_fg]
    fg_rot = rendervar['rotations'][is_fg]

    rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
    rot = build_rotation(rel_rot)
    neighbor_pts = fg_pts[variables["neighbor_indices"]]
    curr_offset = neighbor_pts - fg_pts[:, None]
    curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
    
    loss_rigid = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                            variables["neighbor_weight"])

    loss_l1 = torch.nn.functional.l1_loss(im, batch['im'])
    loss_ssim = (1.0 - calc_ssim(im, batch['im']))
    loss_im = 0.8*loss_l1 + 0.2*loss_ssim

    bg_pts = rendervar['means3D'][~is_fg]
    bg_rot = rendervar['rotations'][~is_fg]
    loss_bg = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

    wandb.log({
            f'loss-l1': loss_l1.item(),
            f'loss-ssim': loss_ssim.item(),
            f'loss-image': loss_im.item(),
            f'loss-rigid': loss_rigid.item(),
            f'loss-bg': loss_bg.item(),
        })

    return loss_im + 3*alpha*loss_rigid + 0*loss_bg

def init_variables(params, num_knn=20):
    variables = {}

    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means'][is_fg]
    init_bg_pts = params['means'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['rotations'][~is_fg])

    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)

    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['rotations']).detach()
    
    return variables

def initialize_per_timestep(params, variables):
    pts = params['means']
    rot = torch.nn.functional.normalize(params['rotations'])

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach().clone()
    variables['prev_offset'] = prev_offset.detach().clone()
    variables["prev_col"] = params['colors'].detach().clone()
    variables["prev_pts"] = pts.detach().clone()
    variables["prev_rot"] = rot.detach().clone()

    return variables

def get_frame(params, batch):
    rendervar = params2rendervar(params)

    im, _, _, = Renderer(raster_settings=batch['cam'])(**rendervar)

    return im

def load_params(path: str):
    params = torch.load(path)

    for v in params.values():
        v.requires_grad = False

    return params

# class MLP(nn.Module):
#     def __init__(self, in_dim, seq_len) -> None:
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(in_dim + 3, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, 512)
#         self.fc4 = nn.Linear(512, 512)
#         self.fc5 = nn.Linear(512, 512)
#         self.fc6 = nn.Linear(512, in_dim)

#         self.relu = nn.ReLU()

#         self.emb = nn.Embedding(seq_len, 3)

#     def dec2bin(self, x, bits):
#         # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
#         mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
#         return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

#     def forward(self, x, t):
#         B, D = x.shape

#         x_ = x

#         e = self.emb(t).repeat(B, 1)
#         # e = self.dec2bin(t, 3).repeat(B, 1)

#         x = torch.cat((x, e), dim=1)
#         # x = x + e

#         x = x # + e
#         x = self.relu(self.fc1(x))
#         x1 = x
#         x = self.relu(self.fc2(x))
#         x2 = x
#         x = self.relu(self.fc3(x))
#         x = self.relu(self.fc4(x))
#         x = x + x2
#         x = self.relu(self.fc5(x))
#         x = x + x1
#         x =           self.fc6(x)

#         return x_ + x

class ResidualBlock(nn.Module):
    def __init__(self, dim) -> None:
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        identity = x

        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.gelu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        
        x += identity
        x = nn.functional.gelu(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, seq_len, block_num) -> None:
        super(MLP, self).__init__()
        
        self.fc_in = nn.Linear(in_dim, hid_dim)
        self.res_blocks = nn.Sequential(
            *(ResidualBlock(hid_dim) for _ in range(block_num))
        )
        self.fc_out = nn.Linear(hid_dim, 7)

        self.emb = nn.Embedding(seq_len, 3)

    def dec2bin(self, x, bits):
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    def forward(self, x, x_, t):
        B, D = x.shape

        identity = x

        # e = self.emb(t).repeat(B, 1)
        # e = self.dec2bin(t, 3).repeat(B, 1)

        x = torch.cat((x_, t), dim=1)
        # x = x + e

        x = self.fc_in(x)
        x = self.res_blocks(x)
        x = self.fc_out(x)

        x += identity

        return x

class UNet(nn.Module):
    def __init__(self, in_dim, hid_dim, seq_len, block_num) -> None:
        super(UNet, self).__init__()
        
        self.fc_in = nn.Linear(in_dim, 512)
        self.fc_1  = nn.Linear(512, 256)
        self.fc_2  = nn.Linear(256, 128)
        self.fc_3  = nn.Linear(128, 64)
        self.fc_4  = nn.Linear(64, 128)
        self.fc_5  = nn.Linear(128, 256)
        self.fc_6  = nn.Linear(256, 512)
        self.fc_out = nn.Linear(512, 7)

        self.emb = nn.Embedding(seq_len, 3)

    def dec2bin(self, x, bits):
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    def forward(self, x, x_, t):
        B, D = x.shape

        identity = x

        # e = self.emb(t).repeat(B, 1)
        # e = self.dec2bin(t, 3).repeat(B, 1)

        # x = torch.cat((x, e), dim=1)
        x = torch.cat((x_, t), dim=1)
        
        x = nn.functional.gelu(self.fc_in(x))
        x_1 = x # 512
        x = nn.functional.gelu(self.fc_1(x))
        x_2 = x # 256
        x = nn.functional.gelu(self.fc_2(x))
        x_3 = x # 128
        x = nn.functional.gelu(self.fc_3(x))
        x = nn.functional.gelu(self.fc_4(x))
        x = x + x_3# 128
        x = nn.functional.gelu(self.fc_5(x))
        x = x + x_2
        x = nn.functional.gelu(self.fc_6(x))
        x = x + x_1
        x = nn.functional.gelu(self.fc_out(x))

        x += identity

        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, L):
        super(PositionalEncoding, self).__init__()
        self.L = L
        self.consts = ((torch.ones(L)*2).pow(torch.arange(L)) * torch.pi).cuda()
    
    def forward(self, x):
        x = x[:,:,None]
        A = (self.consts * x).repeat_interleave(2,2)
        A[:,:,::2] = torch.sin(A[:,:,::2])
        A[:,:,1::2] = torch.cos(A[:,:,::2])

        return A.permute(0,2,1).flatten(start_dim=1)

def get_linear_warmup_cos_annealing(optimizer, warmup_iters, total_iters):
        scheduler_warmup = LinearLR(optimizer, start_factor=1/1000, total_iters=warmup_iters)
        scheduler_cos_decay = CosineAnnealingLR(optimizer, T_max=total_iters-warmup_iters)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, 
                                    scheduler_cos_decay], milestones=[warmup_iters])

        return scheduler

def train(seq: str, iterations, lr, init_cloud, T):
    md = json.load(open(f"./data/{seq}/train_meta.json", 'r'))
    seq_len = T # len(md['fn'])
    params = load_params(init_cloud)
    variables = init_variables(params)
    iterations = iterations #200_000

    mlp = MLP(100, 128, seq_len, 6).cuda()
    # mlp = UNet(100, None, seq_len, None).cuda()
    mlp_optimizer = torch.optim.Adam(params=mlp.parameters(), lr=lr)
    scheduler = get_linear_warmup_cos_annealing(mlp_optimizer, warmup_iters=15_000, total_iters=iterations)

    means = params['means']
    rotations = params['rotations']

    means_norm = means - means.min(dim=0).values
    means_norm = (2. * means_norm / means_norm.max(dim=0).values) - 1.

    rotations_norm = rotations - rotations.min(dim=0).values
    rotations_norm = (2. * rotations_norm / rotations_norm.max(dim=0).values) - 1.

    pos_mean = PositionalEncoding(L=10)
    pos_smol = PositionalEncoding(L=4)

    means_norm = pos_mean(means_norm)
    rotations_norm = pos_smol(rotations_norm)

    ## Random Training
    dataset = []
    for t in range(1, seq_len + 1, 1):
        dataset += [get_dataset(t, md, seq)]
    for i in tqdm(range(iterations)):
        p = i / iterations
        alpha = 2. / (1. + math.exp(-6 * p)) - 1

        di = (i % seq_len)# torch.randint(0, len(dataset), (1,))
        si = torch.randint(0, len(dataset[0]), (1,))

        t = pos_smol(torch.tensor((di+1)/seq_len).view(1,1).repeat(means_norm.shape[0], 1).cuda())

        if di == 0:
            variables = initialize_per_timestep(params, variables)

        X = dataset[di][si]

        # delta = mlp(torch.cat((params['means'], params['rotations']), dim=1), torch.tensor(di).cuda())
        delta = mlp(torch.cat((params['means'], params['rotations']), dim=1), torch.cat((means_norm, rotations_norm), dim=1), torch.tensor(t).cuda())
        delta_means = delta[:,:3]
        delta_rotations = delta[:,3:]

        l = 0.01
        updated_params = copy.deepcopy(params)
        updated_params['means'] = updated_params['means'].detach()
        updated_params['means'] += delta_means * l
        updated_params['rotations'] = updated_params['rotations'].detach()
        updated_params['rotations'] += delta_rotations * l

        loss = get_loss(updated_params, X, variables, alpha)

        variables = initialize_per_timestep(updated_params, variables) # sets previous state to updated state

        wandb.log({
            f'loss-random': loss.item(),
            f'lr': mlp_optimizer.param_groups[0]['lr']
        })

        loss.backward()

        mlp_optimizer.step()
        scheduler.step()
        mlp_optimizer.zero_grad()

    for d in dataset:
        losses = []
        with torch.no_grad():
            for X in d:
                loss = get_loss(updated_params, X, variables, alpha=1.)
                losses.append(loss.item())

        wandb.log({
            f'mean-losses-new': sum(losses) / len(losses)
        })

        ## Visualize
    with torch.no_grad():
        ds = get_dataset(0, md=md, seq=seq)
        canon_batch = ds[0]

        frames = []
        images = []
        gif = []

        # Canonical frame
        frames.append(get_frame(params, canon_batch))

        for t in range(0, seq_len , 1):
            das = get_dataset(t, md=md, seq=seq)
            X = das[0]

            t = pos_smol(torch.tensor((t+1)/seq_len).view(1,1).repeat(means_norm.shape[0], 1).cuda())

            # delta = mlp(torch.cat((params['means'], params['rotations']), dim=1), torch.tensor(t).cuda())
            delta = mlp(torch.cat((params['means'], params['rotations']), dim=1), torch.cat((means_norm, rotations_norm), dim=1), t)
            delta_means = delta[:,:3]
            delta_rotations = delta[:,3:]

            updated_params = copy.deepcopy(params)
            updated_params['means'] = updated_params['means'].detach()
            updated_params['means'] += delta_means * l
            updated_params['rotations'] = updated_params['rotations'].detach()
            updated_params['rotations'] += delta_rotations * l

            fr = get_frame(updated_params, X)

            frames.append(fr)

        for frame in frames:
            frame_np = (frame.detach().cpu().clip(min=0.0, max=1.0).permute(1,2,0).numpy()*255).astype(np.uint8)
            im = Image.fromarray(frame_np)
            images.append(im)

        print('Writing image...')
        images[0].save('temp_result.gif', save_all=True,optimize=False, append_images=images[1:], loop=0)

        print('Saving model...')
        torch.save(mlp.state_dict(), 'model_final.pth')


def main():
    torch.manual_seed(31)

    wandb.login(
        key="45f1e71344c1104de0ce98dc2cf5d9e7557e88ea"
    )
    wandb.init(
        project="new-dynamic-gaussians",
        entity="myasincifci",

    )

    sequence = sys.argv[1]
    iterations = int(sys.argv[2])
    lr = float(sys.argv[3])
    init_cloud = sys.argv[4]
    T = int(sys.argv[5])

    train(sequence, iterations, lr, init_cloud, T)

if __name__ == '__main__':
    main()
