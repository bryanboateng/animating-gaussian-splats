# Animating Gaussian Splats

## Setup

1. Clone this repository

   ```bash
   git clone <repository_url>
   ```
1. Navigate to the repository directory

   ```bash
   cd <repository_directory>
   ```
1. Create a new conda environment using the provided `environment.yml` file:

   ```bash
   conda env create --file environment.yml
   ```
1. Activate the conda environment:

   ```bash
   conda activate animating_gaussian_splats
   ```
1. Install rendering code
	1. Initialize and update the submodules:

		```bash
		git submodule update --init --recursive
		```
   1. Navigate to the `diff-gaussian-rasterization-w-depth` directory:

      ```bash
      cd diff-gaussian-rasterization-w-depth
      ```
   1. Install the package by running the setup script:

      ```bash
      python setup.py install
      ```
   1. Install the package dependencies:

      ```bash
      pip install .
      ```
1. Return to the main repository directory:

   ```bash
   cd ..
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
