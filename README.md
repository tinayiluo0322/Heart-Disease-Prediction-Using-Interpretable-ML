# Heart-Disease-Prediction-Using-Interpretable-ML

## Introduction

In this notebook, I will be exploring a heart disease dataset to analyze and predict the likelihood of heart disease in patients using interpretable machine learning models. The dataset, which comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) and [Kaggle](https://www.kaggle.com/datasets/ketangangal/heart-disease-dataset-uci), contains 1025 observations and 14 features, including attributes such as age, cholesterol levels, and exercise-induced angina. The goal is to predict the presence or absence of heart disease based on these medical indicators.

To build interpretable models for this task, I will be using three distinct models from the `imodels` package, each of which offers interpretability in its own way:
1. **RuleFit**: This model generates a combination of decision tree-based rules (via gradient boosting) and a linear model with Lasso regularization to select the most important rules and features for prediction. It balances model complexity and interpretability by choosing a sparse set of rules that are easy to understand.
   
3. **GreedyRuleListClassifier**: A sequential, rule-based model that creates interpretable decision lists, where each decision path leads to a simple if-then rule. This model is designed to generate compact and interpretable models with high accuracy using a greedy approach.

4. **OptimalTreeClassifier**: This model aims to build optimal decision trees that balance classification accuracy and tree complexity. It has two approaches: GOSDT, which is a globally optimal sparse decision tree algorithm that provides an optimal solution by minimizing error and complexity, and **GreedyTreeClassifier** (CART), which uses a greedy algorithm similar to standard decision trees to find locally optimal splits.

Each of these models will help us not only predict the presence of heart disease but also provide valuable insights into the underlying factors that contribute to the prediction, making the models interpretable and useful in a medical context.
