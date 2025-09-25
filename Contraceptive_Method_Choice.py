import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import label
from six import binary_type
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, roc_curve, auc, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import torch
import torch.nn as nn
import torch.optim as optim
from sympy.integrals.manualintegrate import exp_rule
import joblib

# Import Dataset
data = pd.read_csv('data/cmc.data', header=None, names=['Wife_Age', 'Wife_Education', 'Husband_Education', 'Children_Number', 'Wife_Religion', 'Wife_Working', 'Husband_Occupation', 'Living_Index', 'Media_Exposure', 'Contraceptive_Method'])
X = data.iloc[:, :9]
y = data['Contraceptive_Method']

# Correlation and Heatmap
combined_df = pd.concat([X, y], axis=1)
correlation_matrix = combined_df.corr()
y_correlation = correlation_matrix[['Contraceptive_Method']].drop('Contraceptive_Method')
print('Correlation Map:')
print(y_correlation)
print('---------------------------')
sns.set(style = 'white')
plt.figure(figsize=(9, 9))
heatmap = sns.heatmap(correlation_matrix, annot = True, fmt = ".2f", cmap = 'coolwarm', square= True , linewidths = 0.5, xticklabels = X.columns, yticklabels = X.columns)
plt.title('Heatmap of Contraceptive Method Choice Dataset Features')
plt.savefig('Heatmap of Contraceptive Method Choice.png')
plt.show()

plt.figure(figsize=(9, 9))
sns.heatmap(y_correlation, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap Analysis of Contraceptive Method Choice Dataset Features')
plt.savefig('Correlation Heatmap of Contraceptive Method Choice.png')
plt.show()


# Histograms
for i in range(X.shape[1]):
    plt.figure(figsize=(9, 9))
    sns.histplot(X.iloc[:, i], bins=50, kde=True, alpha=0.7)
    plt.title('Histogram of ' + X.columns[i])
    plt.xlabel(X.columns[i])
    plt.ylabel('Freq')
    plt.savefig('Histogram of ' + X.columns[i])
    plt.show()


# Class Distribution
plt.figure(figsize=(9, 9))
sns.countplot(x = y, alpha=0.7)
plt.title('Histogram of Contraceptive Method Class distribution')
plt.xlabel('Contraceptive_Method')
plt.ylabel('Freq')
plt.savefig('Histogram of Contraceptive Method Class distribution')
plt.show()


labels = ['No-Use', 'Long-Term', 'Short-Term']
plt.figure(figsize=(9, 9))
y.value_counts().plot.pie(autopct = '%1.2f%%', startangle = 90, cmap = 'coolwarm', labels = labels)
plt.title('Pie Chart of Contraceptive Method Class distribution')
plt.xlabel('')
plt.ylabel('')
plt.savefig('Pie Chart of Contraceptive Method Class distribution')
plt.show()

seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)
print('Results of Data Split:')
print(f'Train Data Set: {X_train.shape[0]}')
print(f'Test Data Set: {X_test.shape[0]}')
print('---------------------------')

# tree_model = joblib.load('best_decision_tree_model.pkl')
# y_pred = tree_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='macro', zero_division = 0)
# recall = recall_score(y_test, y_pred, average='macro', zero_division = 0)
# print(f'Tree Depth: {tree_depth}')
# print(f'Accuracy: {accuracy: .3f}')
# print(f'Precision: {precision: .3f}')
# print(f'Recall: {recall: .3f}')
# print('---------------------------')
#
# plt.figure(figsize=(20, 12))
# tree.plot_tree(tree_model, feature_names = X.columns, class_names = y.astype(str), filled=True)
# plt.title("Part B Decision Tree Model")
# plt.savefig('Part B Decision Tree Model.png')
# plt.show()