# Animating Gaussian Splats

## Google Colab Notebooks

For a quick and easy way to test and run the scripts,
you can use the provided Google Colab notebook.
This notebook allows you to execute the code without setting up a local environment.

Google Colab Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bryanboateng/animating-gaussian-splats/blob/main/train.ipynb)

## Prerequisites

To run the scripts locally, ensure you have the following prerequisites:

- Python 3.10 or higher
- Dependencies listed in the `requirements.txt` file

### Setup

1. Clone this repository

   ```bash
   git clone <repository_url>
   ```
1. Navigate to the repository directory

   ```bash
   cd <repository_directory>
   ```
1. Initialize and update the submodules:

   ```bash
   git submodule update --init --recursive
   ```

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Use the following command:

```bash
python -m train sequence_name data_directory_path [options]
```

To see all available arguments and options, run:

```bash
python -m train -h
```
