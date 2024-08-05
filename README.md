# 4D Gaussian Splatting

## Google Colab Notebooks

For a quick and easy way to test and run the scripts,
you can use the provided Google Colab notebooks.
These notebooks allow you to execute the code without setting up a local environment.

| Create                                                                                                                                                                                                                                                            | View                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![Open "Create Deformation Network"-Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bryanboateng/4d-gaussian-splatting/blob/main/deformation_network/google_colab_notebooks/create.ipynb) | [![Open "View Deformation Network" Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bryanboateng/4d-gaussian-splatting/blob/main/deformation_network/google_colab_notebooks/view.ipynb) |

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

### Creating a Deformation Network

To create a deformation network, use the following command:

```bash
python -m create sequence_name data_directory_path [options]
```

To see all available arguments and options, run:

```bash
python -m create -h
```

### Viewing a Deformation Network

To view the sequence represented by the stored deformation network,
use the following command:

```bash
python -m view sequence_name data_directory_path [options]
```

To see all available arguments and options, run:

```bash
python -m view -h
```
