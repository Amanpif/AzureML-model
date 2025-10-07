# 🧠 Azure ML Diabetes Classification Pipeline

A complete **end-to-end MLOps project** built using **Azure Machine Learning v2 SDK**, demonstrating how to orchestrate machine learning workflows with components, pipelines, model tuning, and deployment.

---

## 📂 Project Structure

azureml-diabetes-pipeline/
├── components/
│ ├── preprocessing/
│ │ ├── preprocessing.py # Cleans and prepares the dataset
│ │ └── preprocessing.yaml # Component specification
│ ├── training/
│ │ ├── training.py # Trains ML model (RandomForest / Logistic Regression)
│ │ └── training.yaml # Component specification
│ └── validation/
│ ├── validation.py # Evaluates model performance
│ └── validation.yaml # Component specification
│
├── pipeline/
│ ├── scoring_script.py # Scoring script used during deployment
│ └── pipeline_notebook.ipynb # Orchestrates the entire Azure ML pipeline



## 🧾 Objective

To build, tune, and deploy a machine learning model that predicts whether a patient is **Diabetic** based on health features.  
The project demonstrates MLOps best practices including:
- Modular components (`preprocessing`, `training`, `validation`)
- Pipelines for orchestration
- Model registration and deployment
- Endpoint scoring

---

## ⚙️ Environment Setup

### Prerequisites
- Python ≥ 3.9  
- Azure subscription  
- Azure Machine Learning workspace  
- Azure CLI (≥ 2.40)  
- Azure ML Python SDK v2  
