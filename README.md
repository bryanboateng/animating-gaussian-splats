# 4D Gaussian Splatting

## Google Colab Notebooks

For a quick and easy way to test and run the scripts,
you can use the provided Google Colab notebooks.
These notebooks allow you to execute the code without setting up a local environment.

- Training: [![Open Training Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bryanboateng/4d-gaussian-splatting/blob/main/google_colab_runners/training.ipynb)
- Cloud Video Generation: [![Open Visualization Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bryanboateng/4d-gaussian-splatting/blob/main/google_colab_runners/generate_cloud_video.ipynb)

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
 /content/drive/MyDrive/4d-gaussians/input-data/ \
 basketball \
 --output_directory_path /content/drive/MyDrive/4d-gaussians/output-parameters/ \
```

To see all available options, run:

```bash
python train.py -h
```

### Interactive Cloud Viewer Script

This script visualizes the training data interactively using Open3D.
Note that this script may not work in all environments (e.g., Google Colab).

To run the Interactive Cloud Viewer script:

```bash
python visualizations/view_clouds_interactively.py [options]
```

Example:

```bash
python visualizations/view_clouds_interactively.py \
  --parameters_directory_path /path/to/output-parameters/ \
  --experiment_id foo \
  --sequence_name basketball \
```

To see all available options, run:

```bash
python visualizations/view_clouds_interactively.py -h
```

### Cloud Video Generator Script

If the Interactive Cloud Viewer script is not supported in your environment,
you can use the Cloud Video Generator script, which renders the data as a video.

To run the Cloud Video Generator script:

```bash
python visualizations/generate_cloud_video.py [options]
```

Example:

```bash
python visualizations/generate_cloud_video.py \
  /path/to/output-parameters/ \
  foo \
  basketball \
  --rendered_sequence_directory_path /path/to/renders/ \
  --image_width 1280 \
  --image_height 720 \
  --render_degrees_per_second 90
```

To see all available options, run:

```bash
python visualizations/generate_cloud_video.py -h
```
