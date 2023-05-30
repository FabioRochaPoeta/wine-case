import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
wine = pd.read_csv('winequalityN.csv', sep=",")

# Create the 'opinion' column based on 'quality'
wine['opinion'] = wine['quality'].apply(lambda x: 0 if x <= 5 else 1)

# Drop the 'quality' column
wine.drop('quality', axis=1, inplace=True)

# Describe the variables in the dataset
variables = wine.columns.tolist()
variable_types = wine.dtypes.tolist()
means = wine.mean().tolist()
stds = wine.std().tolist()

# Print variable descriptions
for var, var_type, mean, std in zip(variables, variable_types, means, stds):
    print(f"Variable: {var}\t Type: {var_type}\t Mean: {mean}\t Standard Deviation: {std}")

# Define the features and target variables
X = wine.drop('opinion', axis=1)
y = wine['opinion']

# Define the models
models = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    SVC()
]

# Define the evaluation metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1_score': 'f1'
}

# Perform cross-validation and calculate the evaluation metrics
results = []
for model in models:
    cv_results = cross_validate(model, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True), scoring=scoring)
    results.append(cv_results)

# Calculate the mean and standard deviation of the evaluation metrics
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
mean_scores = []
std_scores = []
for metric in metrics:
    metric_scores = [result[f'test_{metric}'].mean() for result in results]
    mean_score = sum(metric_scores) / len(metric_scores)
    std_score = sum(metric_scores) / len(metric_scores)
    mean_scores.append(mean_score)
    std_scores.append(std_score)

# Print the mean and standard deviation of the evaluation metrics
for metric, mean, std in zip(metrics, mean_scores, std_scores):
    print(f"Metric: {metric}\t Mean: {mean}\t Standard Deviation: {std}")


