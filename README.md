# AI_GIS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

A comprehensive project demonstrating the integration of Artificial Intelligence and Geographic Information Systems (GIS) through practical machine learning applications. This repository serves as an educational resource for researchers and practitioners interested in applying modern AI techniques to geospatial data analysis and remote sensing.

## 🌟 Overview

This repository showcases two comprehensive case studies that illustrate how AI techniques can be effectively applied to spatial and geographic data:

1. **🔢 Parametric Data Analysis and Feature Engineering** - Exploring regression analysis and advanced feature engineering techniques using height-weight datasets and circular spatial patterns
2. **🛰️ CNN-based Super-Resolution** - Implementing deep convolutional neural networks for satellite image super-sampling and enhancement using the UC Merced Land Use dataset

The project emphasizes **practical implementation** of machine learning workflows in geospatial contexts, covering the complete pipeline from data preprocessing and feature engineering to model training, evaluation, and prediction.

> **📊 Data Sources**: All datasets used in this project are obtained from publicly accessible sources. Complete URLs and references for these data sources are provided in the respective loading scripts for full transparency and reproducibility.

## 📁 Project Structure

```
AI_GIS/
├── Case_01_parameters_and_feature/          # 🔢 Parametric analysis and feature engineering
│   ├── step_01_data_processing.py           # Data loading and synthetic data generation
│   ├── step_02_linear_regression.py         # Linear regression modeling
│   ├── step_03_feature_enginering.py        # Advanced feature engineering techniques
│   ├── step_04_machine_learning_linear_regression.py  # ML-enhanced linear regression
│   ├── step_05_machine_learning_circle_separating.py  # Circular pattern classification
│   └── data/                                # Training and test datasets
│       ├── circle_data_decision_line.csv
│       ├── circle_data.csv
│       └── height-weight-20.csv
├── Case_02_CNN_super_sample/                # 🛰️ CNN-based image super-resolution
│   ├── helper.py                            # Utility functions for CNN workflows
│   ├── step_01_get_datasets.py             # UC Merced dataset loading and preprocessing
│   ├── step_02_upsample_use_sklearn.py     # Traditional upsampling methods comparison
│   ├── step_03_upsample_use_CNN_train.py   # CNN model training for super-resolution
│   ├── step_04_upsample_use_CNN_pred_RGB.py # CNN prediction and RGB output generation
│   └── data/                                # Training data and model outputs
│       ├── images/                         # Processed image datasets
│       └── UC_Merced/                     # Original UC Merced dataset
├── installer/                               # Installation utilities and dependencies
├── requirements.yml                         # Conda environment specification
├── LICENSE                                  # MIT License file
└── README.md                               # This documentation
```

## 📚 Detailed Case Study Descriptions

### Case 01: Parameters and Feature Engineering 🔢

This case study demonstrates fundamental machine learning concepts applied to both real-world and synthetic datasets:

- **📥 Data Processing**: Loads height-weight data from online sources and generates parametric circle data with customizable noise parameters for controlled experimentation
- **📈 Linear Regression**: Implements basic regression modeling with comprehensive performance evaluation metrics
- **🔧 Feature Engineering**: Advanced feature creation and transformation techniques for improved model performance and interpretability
- **🤖 Machine Learning Integration**: Enhanced regression models with scikit-learn and advanced classification for circular pattern separation

**Key Technologies**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

**Learning Objectives**:
- Understanding data preprocessing and synthetic data generation
- Implementing regression analysis with performance metrics
- Mastering feature engineering techniques for model improvement
- Applying classification algorithms to spatial pattern recognition

### Case 02: CNN Super-Resolution 🛰️

This case study implements state-of-the-art deep learning approaches for satellite image enhancement:

- **🗃️ Dataset Management**: Utilizes the UC Merced Land Use dataset from Hugging Face, containing 21 distinct land-use scene categories (agricultural, airplane, baseball diamond, etc.)
- **📊 Baseline Comparison**: Implements traditional upsampling methods using Scikit-learn for rigorous performance benchmarking
- **🧠 CNN Architecture**: Custom convolutional neural network specifically designed for image super-resolution tasks with optimized layer configurations
- **🌈 RGB Enhancement**: Generates high-quality RGB outputs from low-resolution satellite imagery with improved spatial detail and color accuracy

**Key Technologies**: PyTorch, Torchvision, Hugging Face Datasets, PIL, XArray, Rasterio, OpenCV

**Learning Objectives**:
- Understanding deep learning architectures for computer vision
- Implementing CNN-based super-resolution techniques
- Working with satellite imagery and remote sensing data
- Comparing traditional vs. deep learning approaches for image enhancement

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (recommended: Python 3.11)
- Mamba or Conda package manager
- Git for version control
- CUDA-compatible GPU (optional, but recommended for CNN training)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JinzhuWANG/AI_GIS.git
   cd AI_GIS
   ```

2. **Install Miniforge** (if not already installed):
   - Download from [miniforge releases](https://github.com/conda-forge/miniforge/releases)
   - Follow the installation instructions for your operating system

3. **Create the conda environment**:
   ```bash
   mamba env create -f requirements.yml
   ```

4. **Activate the environment**:
   ```bash
   conda activate ai_gis
   ```

5. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## 💻 Usage

### Running Case 01: Parameters and Feature Engineering

Execute the scripts in sequence to explore parametric data analysis:

```bash
cd Case_01_parameters_and_feature
python step_01_data_processing.py          # Generate and process datasets
python step_02_linear_regression.py        # Run regression analysis
python step_03_feature_enginering.py       # Apply feature engineering
python step_04_machine_learning_linear_regression.py  # Enhanced ML regression
python step_05_machine_learning_circle_separating.py  # Circular pattern classification
```

**Expected Outputs**: Processed datasets, regression models, feature importance plots, and classification results with performance metrics.

### Running Case 02: CNN Super-Resolution

Execute the CNN pipeline for satellite image enhancement:

```bash
cd Case_02_CNN_super_sample
python step_01_get_datasets.py             # Download and prepare UC Merced dataset
python step_02_upsample_use_sklearn.py     # Compare traditional upsampling methods
python step_03_upsample_use_CNN_train.py   # Train CNN model (may take 30-60 minutes)
python step_04_upsample_use_CNN_pred_RGB.py # Generate enhanced RGB images
```

**Expected Outputs**: Downloaded datasets, trained CNN models, comparison metrics, and enhanced satellite images.

**⚠️ Note**: CNN training in step 3 may take significant time depending on your hardware. Consider using GPU acceleration if available.

## ✨ Key Features

- **🔄 Comprehensive ML Pipeline**: End-to-end workflows from data processing to model deployment and evaluation
- **📊 Multiple Data Types**: Support for both tabular data (height-weight relationships) and high-resolution satellite imagery
- **🎓 Educational Structure**: Step-by-step implementation with clear progression and detailed documentation
- **🔬 Reproducible Results**: Fixed random seeds, version-controlled dependencies, and deterministic algorithms
- **⚡ Modern Tech Stack**: Utilizes latest versions of PyTorch, Hugging Face, and cutting-edge geospatial libraries
- **🌍 Real-world Applications**: Practical examples directly relevant to GIS, remote sensing, and spatial analysis
- **📈 Performance Benchmarking**: Comprehensive comparison between traditional and deep learning approaches
- **🎯 Production-Ready**: Well-structured code with proper error handling and optimization strategies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- 🐛 Bug fixes and performance improvements
- 📚 Additional case studies or practical examples
- 📖 Documentation enhancements and tutorials
- ⚡ Performance optimizations and code refactoring
- 🔧 New feature implementations
- 🧪 Additional test cases and validation methods

### Development Guidelines

1. **Code Structure**: Ensure all code follows the existing project structure and naming conventions
2. **Documentation**: Include clear docstrings, comments, and update README when necessary
3. **Testing**: Test new features with the provided datasets and validate results
4. **Dependencies**: Update `requirements.yml` if new dependencies are added
5. **Git Workflow**: Use descriptive commit messages and create feature branches for new developments

## 📞 Contact

For questions, collaboration opportunities, or technical support, please reach out:

- **Email**: wangjinzhulala@gmail.com
- **GitHub**: [@JinzhuWANG](https://github.com/JinzhuWANG)
- **Issues**: Use GitHub Issues for bug reports and feature requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

*Made with ❤️ for the AI and GIS community*

</div>