# Automatic Uplift Framework (AUF)

AutoML library for automatic uplift modeling

---

## Table of Contents

1. Description
2. Key Features
3. Installation
4. Quick Start
5. Project Structure
6. License

---

## Description

AUF (Automatic Uplift Framework) is an AutoML library that provides a complete pipeline for building uplift models. The library automates all stages: data validation, statistical significance testing of treatment effects, feature selection and ranking, model training (S-, T-, X-learners, uplift trees and forests), optimal model selection, and generation of detailed quality analytics with visualization.

AUF supports multi-treatment and integrates with MLflow for experiment tracking, making it an ideal tool for rapid prototyping and production-ready solutions in personalized interventions.

---

## Key Features

- Complete AutoML pipeline from raw data to production-ready model
- Statistical effect validation via bootstrap significance testing
- Automatic feature selection with 5 possible ranking strategies (filters, importance, permutation, stepwise selection)
- Support for all major uplift methods: S-Learner, X-Learner, uplift trees, random forest
- Comprehensive visualization: Qini curves, uplift curves, conversion by buckets, discrete and continuous plots
- MLflow integration for automatic logging of metrics, artifacts, and models
- Multi-treatment support

---

## Installation

Standard installation from PyPI:

```bash
# Create a new virtual environment (highly recommended)
conda create -n auf_env python=3.8 -y
conda activate auf_env

# Install AUF
pip install auf
```

Installation from source:

```bash
git clone https://github.com/Alfa-Advanced-Analytics/auf.git
cd auf
pip install -e .
```

---

## Quick Start

```python
from auf.pipeline import UpliftPipeline

# Initialize pipeline
pipeline = UpliftPipeline(
    print_doc=False,
    task_name_mlflow='test_auf',
    run_description='Testing AUF library',
)

# Load data with ID, target, treatment, features columns
df = load_your_data()

# Map unified base columns names into user base columns names
base_cols_mapper = {
    'id': "id",
    'treatment': 'treatment',
    'target': 'target',
    'segm': None
}

# Map treatment groups names into unified ones (0 and 1)
treatment_groups_mapper = {
    "control": 0,
    "treatment": 1
}

# Load data in the pipeline
pipeline.load_sample(
    df,
    base_cols_mapper,
    treatment_groups_mapper
)

# Run full pipeline
pipeline.run()

# All results are:
# 1) saved to MLflow (if configured)
# 2) plotted by pipeline during its work
```

Important: DataFrame df must be pre-formatted with column names for ID, target, and treatment specified via mapping dictionaries.

---

## Project Structure

```
auf/  
├── __init__.py  
├── constants/  
│   ├── # Predefined metrics and parameters  
│   ├── __init__.py  
│   ├── metrics.py  
│   └── numbers.py  
├── data/  
│   ├── # Data validation and preprocessing  
│   ├── __init__.py  
│   ├── checks.py  
│   ├── preprocessing.py  
│   └── split.py  
├── feature_rankers/  
│   ├── # Feature ranking strategies  
│   ├── __init__.py  
│   ├── filter.py  
│   ├── importance.py  
│   ├── permutation.py  
│   ├── stepwise.py  
│   └── straightforward.py  
├── log/  
│   ├── # Logging and progress tracking  
│   ├── __init__.py  
│   └── log.py  
├── metrics/  
│   ├── # Custom uplift metrics  
│   ├── __init__.py  
│   ├── averaged.py  
│   ├── by_top.py  
│   └── overfit.py  
├── ml_flow/  
│   ├── # MLflow integration  
│   ├── __init__.py  
│   └── ml_flow.py  
├── models/  
│   ├── # Uplift model implementations  
│   ├── __init__.py  
│   ├── auf_forest.py  
│   ├── auf_model.py  
│   ├── auf_tree.py  
│   └── auf_x_learner.py  
├── pipeline/  
│   ├── # Main pipeline and components  
│   ├── __init__.py  
│   ├── calibration.py  
│   ├── evaluation.py  
│   ├── inference.py  
│   └── pipeline.py  
├── plots/  
│   ├── # Result visualization  
│   ├── __init__.py  
│   └── plots.py  
└── training/  
    ├── # Training and optimization  
    ├── __init__.py  
    ├── fitting.py  
    ├── gridsearch.py  
    └── model_generation.py  
```

## License

This project is licensed under the MIT License. See the [LICENSE.txt](LICENSE.txt) file for details.