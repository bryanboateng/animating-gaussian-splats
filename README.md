# 4D Gaussian Splatting

## Google Colab Notebooks

For a quick and easy way to test and run the scripts,
you can use the provided Google Colab notebooks.
These notebooks allow you to execute the code without setting up a local environment.

| Notebook Name | Google Colab Link |
| -------------- | --------------- |
| Train Per Timestamp | [![Open "Train Per Timestamp" Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bryanboateng/4d-gaussian-splatting/blob/main/google_colab_runners/train_per_timestamp.ipynb) |
| Train Deformation Network | [![Open "Train Deformation Network" Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bryanboateng/4d-gaussian-splatting/blob/main/google_colab_runners/train_deformation_network.ipynb) |
| Generate Cloud Video | [![Open "Generate Cloud Video" Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bryanboateng/4d-gaussian-splatting/blob/main/google_colab_runners/generate_cloud_video.ipynb)|

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

### Training Scripts

#### "Train Per Timestamp" Script

The "Train Per Timestamp" script is used to train the model by optimizing the Gaussian clouds for each timestamp.

To run the "Train Per Timestamp" script:

```bash
python train_per_timestamp.py sequence_name data_directory_path [options]
```

Example:

```bash
python train_per_timestamp.py \
 basketball \
 /content/drive/MyDrive/4d-gaussians/input-data/ \
 --output_directory_path /content/drive/MyDrive/4d-gaussians/output-parameters/
```

To see all available arguments and options, run:

```bash
python train_per_timestamp.py -h
```

#### "Train Deformation Network" Script

The "Train Deformation Network" script is used to train the model by learning the deformation between timestamps.

To run the "Train Deformation Network" script:

```bash
python train_deformation_network.py sequence_name data_directory_path [options]
```

Example:

```bash
python train_deformation_network.py \
 basketball \
 /content/drive/MyDrive/4d-gaussians/input-data/ \
 --learning_rate 0.01
```

To see all available arguments and options, run:

```bash
python train_deformation_network.py -h
```

### Visualization Scripts

#### "View Clouds Interactively" Script

This script visualizes the training data interactively using Open3D.
> [!NOTE]
> This script may not work in all environments (e.g., Google Colab).

To run the "View Clouds Interactively" script:

```bash
python view_clouds_interactively.py [options]
```

Example:

```bash
python view_clouds_interactively.py \
  --parameters_directory_path /path/to/output-parameters/ \
  --experiment_id foo \
  --sequence_name basketball \
```

To see all available arguments and options, run:

```bash
python view_clouds_interactively.py -h
```

#### "Generate Cloud Video" Script

If the "View Clouds Interactively" script is not supported in your environment,
you can use the "Generate Cloud Video" script, which renders the data as a video.

To run the "Generate Cloud Video" script:

```bash
python generate_cloud_video.py experiment_id sequence_name parameters_directory_path [options]
```

Example:

```bash
python generate_cloud_video.py \
  foo \
  basketball \
  /path/to/output-parameters/ \
  --rendered_sequence_directory_path /path/to/renders/ \
  --image_width 1280 \
  --image_height 720 \
  --render_degrees_per_second 90
```

To see all available arguments and options, run:

```bash
python generate_cloud_video.py -h
```
