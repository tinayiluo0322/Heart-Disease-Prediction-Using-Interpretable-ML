# Heart-Disease-Prediction-Using-Interpretable-ML

## Introduction

In this notebook, we will be exploring a heart disease dataset to analyze and predict the likelihood of heart disease in patients using interpretable machine learning models. The dataset, which comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) and [Kaggle](https://www.kaggle.com/datasets/ketangangal/heart-disease-dataset-uci), contains 1025 observations and 14 features, including attributes such as age, cholesterol levels, and exercise-induced angina. The goal is to predict the presence or absence of heart disease based on these medical indicators.

To build interpretable models for this task, we will be using three distinct models from the `imodels` package, each of which offers interpretability in its own way. I created diagrams for each of the model to help explain the model inner structures:
1. **RuleFit**: This model generates a combination of decision tree-based rules (via gradient boosting) and a linear model with Lasso regularization to select the most important rules and features for prediction. It balances model complexity and interpretability by choosing a sparse set of rules that are easy to understand.

<img width="595" alt="Screen Shot 2024-09-25 at 3 57 46 PM" src="https://github.com/user-attachments/assets/95592965-28ec-47cb-8b8b-a14d25914ab3">
   
2. **GreedyRuleListClassifier**: A sequential, rule-based model that creates interpretable decision lists, where each decision path leads to a simple if-then rule. This model is designed to generate compact and interpretable models with high accuracy using a greedy approach.

<img width="589" alt="Screen Shot 2024-09-25 at 4 12 32 PM" src="https://github.com/user-attachments/assets/b550d577-1324-4c67-8f05-cfee8643a07e">

3. **OptimalTreeClassifier**: This model aims to build optimal decision trees that balance classification accuracy and tree complexity. It has two approaches:
- **GOSDT**: A globally optimal sparse decision tree algorithm that minimizes both error and complexity, providing an optimal solution.
- **GreedyTreeClassifier (CART)**: A greedy algorithm similar to standard decision trees, which finds locally optimal splits to build the tree.
  
<img width="597" alt="Screen Shot 2024-09-25 at 5 12 11 PM" src="https://github.com/user-attachments/assets/cd8b9c06-f850-4658-bd0a-6c4fd2881b20">

Even though the OptimalTreeClassifier includes the GOSDT approach, at the end of the notebook, I also implemented GOSDT through a standalone Python package for more direct control over the model configuration and performance. This allows for a more thorough exploration of GOSDT's capabilities in generating interpretable decision trees.

Each of these models will help us not only predict the presence of heart disease but also provide valuable insights into the underlying factors contributing to the prediction, making the models interpretable and useful in a medical context.

