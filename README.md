# 4D Gaussian Splatting

## Google Colab Notebooks

For a quick and easy way to test and run the scripts,
you can use the provided Google Colab notebooks.
These notebooks allow you to execute the code without setting up a local environment.

- Training: [![Open Training Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bryanboateng/4d-gaussian-splatting/blob/main/google_colab_runners/training.ipynb)
- 2D visualization: [![Open Visualization Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bryanboateng/4d-gaussian-splatting/blob/main/google_colab_runners/visualize_2d.ipynb)

## Prerequisites

To run the scripts locally, ensure you have the following prerequisites:

- Python 3.10 or higher
- Dependencies listed in the `requirements.txt` file

### Setup

1. Clone this repository
1. Navigate to the repository directory
1. Initialize and update the submodules:

   ```bash
   git submodule update --init --recursive
   ```

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Scripts

### Training Script

The training script is used to train the model and save the parameters.

To run the training script:

```bash
python train.py [options]
```

Example:

```bash
python train.py \
 --data_directory_path /content/drive/MyDrive/4d-gaussians/input-data/ \
 --output_directory_path /content/drive/MyDrive/4d-gaussians/output-parameters/ \
 --sequence_name basketball
```

To see all available options, run:

```bash
python train.py -h
```

### 3D Visualization Script

This script visualizes the training data in 3D using Open3D.
Note that this script may not work in all environments (e.g., Google Colab).

To run the 3D visualization script:

```bash
python visualize_3d.py [options]
```

Example:

```bash
python visualize_3d.py \
  --parameters_directory_path /path/to/output-parameters/ \
  --experiment_id foo \
  --sequence_name basketball \
```

To see all available options, run:

```bash
python visualize_3d.py -h
```

### 2D Visualization Script

If the 3D visualization script is not supported in your environment,
you can use the 2D visualization script, which renders the data as a video.

To run the 2D visualization script:

```bash
python visualize_2d.py [options]
```

Example:

```bash
python visualize_2d.py \
  --parameters_directory_path /path/to/output-parameters/ \
  --rendered_sequence_directory_path /path/to/renders/ \
  --experiment_id foo \
  --sequence_name basketball \
  --image_width 1280 \
  --image_height 720 \
  --render_degrees_per_second 90
```

To see all available options, run:

```bash
python visualize_2d.py -h
```
