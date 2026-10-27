# ML_OPS_Demo1
Intro to MLOps

This project demonstrates an end-to-end machine learning workflow using
Python, Scikit-learn, MLflow, and Git/GitHub.

The project uses the Breast Cancer Wisconsin Diagnostic dataset to build
a binary classification model using Logistic Regression.

The main objective is to demonstrate how machine learning experiments
can be tracked, compared, and reproduced using MLflow.

---

## Objectives

- Load and prepare a machine learning dataset
- Perform train-test splitting
- Build a Scikit-learn machine learning pipeline
- Train a Logistic Regression model
- Evaluate the model using multiple metrics
- Track parameters and metrics using MLflow
- Store trained models as MLflow artifacts
- Run multiple experiments with different hyperparameters
- Compare experiments using the MLflow UI
- Version-control the project using Git and GitHub

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- MLflow
- Git
- GitHub

---

## Dataset

The project uses the Breast Cancer Wisconsin Diagnostic dataset
available through Scikit-learn.

The dataset contains:

- 569 samples
- 30 input features
- 1 target variable
- 2 target classes

The dataset is generated and saved locally as:

```text
data/breast_cancer.csv
````

---

## Project Structure

```text
ML_FLOW/
│
├── data/
│   └── breast_cancer.csv
│
├── src/
│   ├── data_collection.py
│   └── train.py
│
├── myenv/
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

The `mlruns/` directory is generated automatically by MLflow and should
not be committed to Git.

---



### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ML_FLOW
```

### 2. Create and activate the virtual environment

For Windows PowerShell:

```powershell
python -m venv myenv
```

Activate it:

```powershell
.\myenv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 1: Generate the Dataset

Run the data collection script:

```bash
python src/data_collection.py
```

This downloads the Breast Cancer dataset using Scikit-learn and saves
it as:

```text
data/breast_cancer.csv
```

---

## Step 2: Train the Model

Run the training script:

```bash
python src/train.py
```

The script performs the following steps:

1. Loads the dataset
2. Separates features and target
3. Splits the data into training and testing sets
4. Standardizes the features
5. Trains a Logistic Regression model
6. Generates predictions
7. Calculates evaluation metrics
8. Logs parameters to MLflow
9. Logs metrics to MLflow
10. Logs the trained model to MLflow

---

## Model

The project uses a Scikit-learn Pipeline:

```text
Input Data
    ↓
StandardScaler
    ↓
Logistic Regression
    ↓
Prediction
```

The Logistic Regression model uses the following configurable parameters:

* `C`
* `max_iter`
* `test_size`

---

## Evaluation Metrics

The following metrics are calculated:

### Accuracy

Measures the percentage of correctly classified observations.

### Precision

Measures how many of the observations predicted as positive
are actually positive.

### Recall

Measures how many of the actual positive observations were correctly
identified.

### F1 Score

Provides a balance between precision and recall.

---

## MLflow Experiment Tracking

The experiment is created using:

```python
mlflow.set_experiment("Breast Cancer Classification")
```

Each model training execution creates a separate MLflow run.

The following parameters are tracked:

```text
model
C
max_iter
test_size
random_state
```

The following metrics are tracked:

```text
accuracy
precision
recall
f1_score
```

The trained Scikit-learn model is also stored as an MLflow artifact.

---

## Running Multiple Experiments

Different values of the Logistic Regression `C` parameter can be tested.

### Experiment 1

```bash
python src/train.py --C 0.01
```

### Experiment 2

```bash
python src/train.py --C 0.1
```

### Experiment 3

```bash
python src/train.py --C 1
```

### Experiment 4

```bash
python src/train.py --C 10
```

Each command creates a separate MLflow run.

This allows the different model configurations and their performance
to be compared.

---

## MLflow UI

Start the MLflow tracking interface:

```bash
mlflow ui
```

MLflow will provide a local URL, typically:

```text
http://127.0.0.1:5000
```

Open this URL in a browser.

The MLflow UI can be used to:

* View experiments
* Compare different runs
* View parameters
* View evaluation metrics
* View model artifacts
* Identify the best-performing run

---

## Experiment Comparison

The experiments can be compared using:

| Parameter | Description                                  |
| --------- | -------------------------------------------- |
| C         | Logistic Regression regularization parameter |
| Accuracy  | Overall classification accuracy              |
| Precision | Positive prediction quality                  |
| Recall    | Positive class detection                     |
| F1 Score  | Balance between precision and recall         |

The best model should be selected based on the evaluation metric that
is most appropriate for the business or problem context.

For this project, F1 Score can be used as the primary comparison metric.

---

## Git Workflow

The project uses Git for version control.

Check the current status:

```bash
git init
git status
```

Add changes:

```bash
git add .
```

```bash
git config --global user.name
git config --global user.email
```

Commit changes:

```bash
git commit -m "Add MLflow experiment tracking"
```

Push changes to GitHub:

```bash
git push origin main
```

View commit history:

```bash
git log --oneline
```

---

## End-to-End Workflow

```text
GitHub Repository
       ↓
Local Git Repository
       ↓
Python Environment
       ↓
Data Collection
       ↓
breast_cancer.csv
       ↓
Data Preparation
       ↓
Train-Test Split
       ↓
Feature Scaling
       ↓
Logistic Regression
       ↓
Model Evaluation
       ↓
MLflow Tracking
       ↓
Multiple Experiments
       ↓
Compare Runs
       ↓
Select Best Model
       ↓
Git Commit
       ↓
GitHub
```

---

## Key Learning Outcomes

After completing this project, you should understand:

* How to structure a basic ML project
* How to use Git for ML project version control
* How to create reproducible Python environments
* How to track ML parameters using MLflow
* How to track ML evaluation metrics
* How to store trained models as MLflow artifacts
* How to run multiple ML experiments
* How to compare experiments using MLflow UI
* How Git and MLflow complement each other in an ML workflow

---

## Future Improvements

The project can be extended by:

* Adding additional classification algorithms
* Performing hyperparameter tuning
* Adding confusion matrix visualization
* Tracking datasets and model versions
* Adding MLflow Model Registry
* Adding automated testing
* Creating a CI/CD pipeline
* Deploying the selected model as an API
* Containerizing the application using Docker

````

