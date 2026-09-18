# Predictive Lab

A collection of machine learning projects focused on prediction, model tuning, and practical GitHub workflows.

## GitHub Basics

### Create a repository

Create an empty repository on GitHub, then connect this local project and upload the code:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<username>/<repository>.git
git push -u origin main
```

### Create and upload a branch

Use a separate branch for each feature or project:

```bash
git switch -c feature/student-performance-prediction
git add .
git commit -m "Add student performance prediction"
git push -u origin feature/student-performance-prediction
```

After pushing, open a pull request on GitHub to merge the branch into `main`.

## Projects

1. **Student Performance Prediction Using Model Tuning**
	- Predict student performance and compare tuned regression or classification models.

2. **House Price Prediction Using Hyperparameter Tuning**
	- Estimate house prices and optimize the model with systematic hyperparameter search.

3. **Student Admission Prediction Using Machine Learning**
	- Predict admission outcomes from academic and application-related features.

4. **Credit Card Fraud Detection Using Machine Learning**
	- Identify potentially fraudulent transactions while accounting for class imbalance.

5. **Customer Churn Prediction**
	- Predict which customers are likely to leave and identify the factors associated with churn.

## Suggested Project Structure

```text
project-name/
├── README.md
├── data/
├── notebooks/
├── src/
├── models/
└── requirements.txt
```

Each project should document its dataset, preprocessing steps, model choices, tuning method, evaluation metrics, and conclusions.