# AI_GIS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 13+](https://img.shields.io/badge/python-13%2B-blue.svg)](https://www.python.org/downloads/)

A comprehensive project demonstrating the integration of artificial intelligence and geographic information systems through practical machine learning applications.

## Overview

This repository showcases two main case studies that illustrate how AI techniques can be applied to spatial and geographic data:

1. **Parametric Data Analysis and Feature Engineering** - Exploring regression analysis and feature engineering techniques using height-weight data and circular spatial patterns
2. **CNN-based Super-Resolution** - Implementing convolutional neural networks for satellite image super-sampling and enhancement using the UC Merced Land Use dataset

The project emphasizes practical implementation of machine learning workflows in geospatial contexts, from data preprocessing to model training and prediction.

**Data Sources**: All data used in this project are obtained from publicly accessible sources. The URLs and references for these data sources are presented when loading the data in the respective scripts.

## Project Structure

```
AI_GIS/
├── Case_01_parameters_and_feature/          # Parametric analysis and feature engineering
│   ├── step_01_data_processing.py           # Data loading and synthetic data generation
│   ├── step_02_linear_regression.py         # Linear regression modeling
│   └── step_03_feature_enginering.py        # Advanced feature engineering techniques
├── Case_02_CNN_super_sample/                # CNN-based image super-resolution
│   ├── helper.py                            # Utility functions for CNN workflows
│   ├── step_01_get_datasets.py             # UC Merced dataset loading and preprocessing
│   ├── step_02_upsample_use_sklearn.py     # Traditional upsampling methods comparison
│   ├── step_03_upsample_use_CNN_train.py   # CNN model training for super-resolution
│   └── step_04_upsample_use_CNN_pred_RGB.py # CNN prediction and RGB output generation
├── installer/                               # Development environment installers
│   ├── Miniforge3-Windows-x86_64.exe       # Conda package manager installer
│   └── VSCodeUserSetup-x64-1.103.1.exe     # Visual Studio Code installer
├── requirements.yml                         # Conda environment specification
├── LICENSE                                  # MIT License file
└── README.md                               # Project documentation
```

## Detailed Case Study Descriptions

### Case 01: Parameters and Feature Engineering

This case study demonstrates fundamental machine learning concepts applied to both real-world and synthetic datasets:

- **Data Processing**: Loads height-weight data from online sources and generates parametric circle data with customizable noise parameters
- **Linear Regression**: Implements basic regression modeling with performance evaluation
- **Feature Engineering**: Advanced feature creation and transformation techniques for improved model performance

**Key Technologies**: NumPy, Pandas, Scikit-learn, Seaborn

### Case 02: CNN Super-Resolution

This case study implements deep learning approaches for satellite image enhancement:

- **Dataset Management**: Utilizes the UC Merced Land Use dataset from Hugging Face, containing 21 land-use scene categories
- **Baseline Comparison**: Implements traditional upsampling methods using Scikit-learn for performance benchmarking
- **CNN Architecture**: Custom convolutional neural network designed for image super-resolution tasks
- **RGB Enhancement**: Generates high-quality RGB outputs from low-resolution satellite imagery

**Key Technologies**: PyTorch, Torchvision, Hugging Face Datasets, PIL, XArray, Rasterio

## Contact

For questions or collaboration opportunities, please contact: wangjinzhulala@gmail.com

## Getting Started

### Prerequisites

- Python 13+
- Mamba package manager

### Installation

1. **Install Miniforge** (if not already installed):
   - Windows users can use the installer provided in `installer/Miniforge3-Windows-x86_64.exe`
   - Or download from [miniforge releases](https://github.com/conda-forge/miniforge/releases)

2. **Create the conda environment**:
   ```bash
   mamba env create -f requirements.yml
   ```

3. **Activate the environment**:
   ```bash
   conda activate ai_gis
   ```

## Usage

### Running Case 01: Parameters and Feature Engineering

Execute the scripts in sequence to explore parametric data analysis:

```bash
cd Case_01_parameters_and_feature
python step_01_data_processing.py      # Generate and process datasets
python step_02_linear_regression.py    # Run regression analysis
python step_03_feature_enginering.py   # Apply feature engineering
```

### Running Case 02: CNN Super-Resolution

Execute the CNN pipeline for satellite image enhancement:

```bash
cd Case_02_CNN_super_sample
python step_01_get_datasets.py         # Download and prepare UC Merced dataset
python step_02_upsample_use_sklearn.py # Compare traditional upsampling methods
python step_03_upsample_use_CNN_train.py # Train CNN model
python step_04_upsample_use_CNN_pred_RGB.py # Generate enhanced RGB images
```

## Key Features

- **Comprehensive ML Pipeline**: End-to-end workflows from data processing to model deployment
- **Multiple Data Types**: Support for both tabular data and satellite imagery
- **Educational Structure**: Step-by-step implementation with clear progression
- **Reproducible Results**: Fixed random seeds and version-controlled dependencies
- **Modern Tech Stack**: Utilizes latest versions of PyTorch, Hugging Face, and geospatial libraries
- **Real-world Applications**: Practical examples relevant to GIS and remote sensing

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes and improvements
- Additional case studies or examples
- Documentation enhancements
- Performance optimizations

### Development Guidelines

1. Ensure all code follows the existing project structure
2. Include clear documentation and comments
3. Test new features with the provided datasets
4. Update requirements.yml if new dependencies are added

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.