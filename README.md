# Heart Disease MLOps Assignment

End-to-end MLOps project for predicting heart disease risk using the UCI Heart Disease dataset.

## Project Status

This repository is being built step by step.

Current step:
- Data acquisition script added

Next step:
- Run data acquisition and start exploratory data analysis

## Planned Workflow

1. Data acquisition and exploratory data analysis
2. Data preprocessing and feature engineering
3. Model training and evaluation
4. Experiment tracking with MLflow
5. Model packaging and reproducibility
6. FastAPI prediction service
7. Unit tests and CI/CD
8. Docker containerization
9. Kubernetes deployment
10. Monitoring, logging, and final report

## Repository Structure

```text
data/                  Dataset files and processed data
notebooks/             EDA and experimentation notebooks
src/                   Training, preprocessing, and inference code
api/                   FastAPI application
models/                Saved trained model artifacts
tests/                 Unit tests
k8s/                   Kubernetes manifests
screenshots/           Evidence for report
reports/               Final report files
.github/workflows/     CI/CD workflows
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Acquisition

Download, clean, and save the UCI Cleveland Heart Disease dataset:

```bash
python -m src.data_processing
```

This creates:

```text
data/raw/heart.csv
data/processed/heart_processed.csv
```
