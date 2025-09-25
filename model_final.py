import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb

# Part A
# Import Dataset
data = pd.read_csv('data/abalone.data', header=None,
                   names=['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
                          'Shell_weight', 'Rings'])
X = data.iloc[:, :8]
y = data['Rings']

# Data cleaning
X['Sex'] = X['Sex'].map({'M': 0, 'F': 1, 'I': 2})

# Checking whether there are values lower than 0
count_error_list = {}
for column in X.columns:
    count_error = (X[column] < 0).sum()
    if count_error > 0:
        count_error_list[column] = count_error
for col, count in count_error_list.items():
    print(f"{col}: {count}")

# Drop empty Value
X = X.dropna()
y = y.iloc[X.index]

# A3Q1 Correlation and Heatmap of Features Distribution
combined_df = pd.concat([X, y], axis=1)
correlation_matrix = combined_df.corr()
y_correlation = correlation_matrix
print('Correlation Map:')
print(y_correlation)
print('---------------------------')
plt.figure(figsize=(9, 9))
sns.heatmap(y_correlation, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap Analysis of Abalone Dataset Features')
plt.savefig('heatmap.png')
plt.show()

# A3Q1 Scatter Plot of Features Distribution
# 1st
scatter_data = pd.DataFrame({'Shell_weight': X['Shell_weight'], 'Rings': y})
plt.figure(figsize=(9, 9))
sns.scatterplot(data=scatter_data, x='Shell_weight', y='Rings', alpha=0.7)
plt.title('Scatter Plots of Features Most Correlated with Ring Age')
plt.xlabel('Shell_weight')
plt.ylabel('Rings')
plt.savefig('Scatter_fig_1st.png')
plt.show()

# 2nd
scatter_data = pd.DataFrame({'Diameter': X['Diameter'], 'Rings': y})
plt.figure(figsize=(9, 9))
sns.scatterplot(data=scatter_data, x='Diameter', y='Rings', alpha=0.7)
plt.title('Scatter Plots of Features 2nd Correlated with Ring Age')
plt.xlabel('Diameter')
plt.ylabel('Rings')
plt.savefig('Scatter_fig_2nd.png')
plt.show()

# A3Q1 Histograms of Features Distribution
plt.figure(figsize=(9, 9))
sns.histplot(X['Shell_weight'], bins=50, kde=True, alpha=0.7)
plt.title('Histogram of Shell Weight')
plt.xlabel('Shell_Weight')
plt.ylabel('Freq')
plt.savefig('Histogram of Shell Weight.png')
plt.show()

plt.figure(figsize=(9, 9))
sns.histplot(X['Diameter'], bins=50, kde=True, alpha=0.7)
plt.title('Histogram of Diameter')
plt.xlabel('Diameter')
plt.ylabel('Freq')
plt.savefig('Histogram of Diameter.png')
plt.show()

plt.figure(figsize=(9, 9))
sns.histplot(y, bins=30, kde=True, alpha=0.7)
plt.title('Histogram of Rings')
plt.xlabel('Rings')
plt.ylabel('Freq')
plt.savefig('Histogram of Rings.png')
plt.show()

# A3Q1 Histogram of Class Distribution
y_log = pd.cut(y, bins=[0, 7, 10, 15, np.inf], labels=[1, 2, 3, 4]).astype(int)
plt.figure(figsize=(9, 9))
sns.countplot(x=y_log, alpha=0.7)
plt.title('Histogram of Class Distribution')
plt.xlabel('Class')
plt.ylabel('Freq')
plt.savefig('Histogram of Class Distribution.png')
plt.show()

# A3Q1 Pie Chart of Class Distribution
labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
plt.figure(figsize=(9, 9))
y_log.value_counts().plot.pie(autopct='%1.2f%%', startangle=90, cmap='coolwarm', labels=labels)
plt.title('Pie Chart of Class Distribution')
plt.xlabel('')
plt.ylabel('')
plt.savefig('Pie Chart of Class Distribution.png')
plt.show()

model_scores = {}

# A3Q2 Decision Tree
tree_depths = list(range(2, 8, 1))
tree_accuracy = []
best_accuracy = 0
y_dt = pd.cut(y, bins=[0, 7, 10, 15, np.inf], labels=[1, 2, 3, 4]).astype(int)
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X, y_dt, test_size=0.4, random_state=10)

for tree_depth in tree_depths:
    order = np.random.permutation(len(X))
    index = int(0.6 * len(X))
    X_train_dt = X.iloc[order[: index]]
    X_test_dt = X.iloc[order[index:]]
    y_train_dt = y_dt.iloc[order[: index]]
    y_test_dt = y_dt.iloc[order[index:]]

    model = DecisionTreeClassifier(random_state=0, max_depth=tree_depth)
    model.fit(X_train_dt, y_train_dt)

    y_pred_dt = model.predict(X_test_dt)

    accuracy = accuracy_score(y_test_dt, y_pred_dt)
    precision = precision_score(y_test_dt, y_pred_dt, average='macro', zero_division=0)
    recall = recall_score(y_test_dt, y_pred_dt, average='macro', zero_division=0)
    f1_tree = f1_score(y_test_dt, y_pred_dt, average='macro')

    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        best_tree_depth = tree_depth
        best_model = model
        model_scores['Decision Tree'] = f1_tree
    print(f'Tree Depth: {tree_depth}')
    print(f'Accuracy: {accuracy: .3f}')
    print(f'Precision: {precision: .3f}')
    print(f'Recall: {recall: .3f}')
    print('---------------------------')
    tree_accuracy.append(accuracy)

print(f'Best Accuracy: {best_accuracy: .3f}')
print(f'Best Tree Depth: {best_tree_depth}')
print('---------------------------')

plt.figure(figsize=(9, 9))
plt.plot(tree_depths, tree_accuracy, marker='o', linestyle='-')
plt.title('Accuracies of Different Tree Depths of Decision Trees')
plt.xlabel('The Tree Depths of Decision Trees')
plt.ylabel('Accuracy')
plt.savefig('Accuracies of Different Tree Depths of Decision Trees.png')
plt.show()

y_pred_best = best_model.predict(X_test_tree)
plt.figure(figsize=(20, 12))
tree.plot_tree(best_model, feature_names=X.columns, class_names=y_dt.astype(str), filled=True)
plt.title("Best Decision Tree Model")
plt.savefig('Best Decision Tree Model.png')
plt.show()

# IF_THEN Rules
rules_dt = export_text(best_model, feature_names=X.columns)
print(rules_dt)
print('---------------------------')

condition = []
lines = rules_dt.splitlines()
with open("rules_output.txt", "w") as f:
    for line in lines:
        depth = line.count('|')
        line = line.replace('|', '').replace('-', '').strip()
        if 'class' in line:
            rule = "IF " + " AND ".join(condition) + " THEN " + line.strip()
            print(rule)
            f.write(rule + '\n')
            while condition and len(condition) > depth:
                condition.pop()
        else:
            while condition and len(condition) >= depth:
                condition.pop()
            condition.append(line.strip())
print('---------------------------')

# A3Q3 Post-Pruning
path = best_model.cost_complexity_pruning_path(X_train_tree, y_train_tree)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

post_pruning_accuracy = []
best_accuracy_pru = 0

for ccp_alpha in ccp_alphas:
    post_pruning_model = DecisionTreeClassifier(random_state=0, max_depth=best_tree_depth, ccp_alpha=ccp_alpha)
    post_pruning_model.fit(X_train_tree, y_train_tree)
    y_pred_tree = post_pruning_model.predict(X_test_tree)
    accuracy = accuracy_score(y_test_tree, y_pred_tree)
    f1_post_tree = f1_score(y_test_tree, y_pred_tree, average='macro')
    print(f'Decision Tree Alpha: {ccp_alpha: .3f}')
    print(f'Accuracy: {accuracy: .3f}')
    print('---------------------------')
    post_pruning_accuracy.append(accuracy)
    if accuracy >= best_accuracy_pru:
        best_accuracy_pru = accuracy
        best_alpha = ccp_alpha
        best_model_pru = post_pruning_model
        model_scores['Post-Pruning Decision Tree'] = f1_post_tree
print(f'Best Alpha: {best_alpha: .3f}')
print(f'Best Accuracy: {best_accuracy_pru: .3f}')
print('---------------------------')

plt.figure(figsize=(9, 9))
plt.plot(ccp_alphas, post_pruning_accuracy, marker='o', linestyle='-')
plt.title('Accuracies of Different Alpha of Decision Tree')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.savefig('Accuracies of Different Alpha of Decision Tree.png')
plt.show()

plt.figure(figsize=(20, 12))
tree.plot_tree(best_model_pru, feature_names=X.columns, class_names=y_dt.astype(str), filled=True)
plt.title("Best Decision Tree Model after Post-Pruning")
plt.savefig('Best Decision Tree Model after Post-Pruning.png')
plt.show()

# A3Q4 Random Forests
tree_count = list(range(1200, 1310, 100))
random_forest_accuracy = []
best_accuracy_rf = 0

for count in tree_count:
    random_tree_model = RandomForestClassifier(n_estimators=count, random_state=0)
    random_tree_model.fit(X_train_tree, y_train_tree)
    y_pred_rf = random_tree_model.predict(X_test_tree)
    accuracy = accuracy_score(y_test_tree, y_pred_rf)
    f1_random = f1_score(y_test_tree, y_pred_rf, average='macro')
    print(f'Random Forests')
    print(f'The Number of Decision Trees: {count}')
    print(f'Accuracy: {accuracy: .3f}')
    print('---------------------------')
    random_forest_accuracy.append(accuracy)
    if accuracy >= best_accuracy_rf:
        best_accuracy_rf = accuracy
        best_count_random_forest = count
        model_scores['Random Forests'] = f1_random
print(f'Best Number of Decision Trees: {best_count_random_forest}')
print(f'Best Accuracy: {best_accuracy_rf: .3f}')
print('---------------------------')

plt.figure(figsize=(9, 9))
plt.plot(tree_count, random_forest_accuracy, marker='o', linestyle='-')
plt.title('Accuracies of Different Numbers of Decision Trees')
plt.xlabel('The Number of Decision Trees')
plt.ylabel('Accuracy')
plt.savefig('Accuracies of Different Numbers of Decision Trees.png')
plt.show()

# A3Q5 XGBoost
tree_count = list(range(300, 401, 100))
xgboost_accuracy = []
best_accuracy_xb = 0
y_train_xg = y_train_tree - 1
y_test_xg = y_test_tree - 1

for count in tree_count:
    xgboost_model = xgb.XGBClassifier(n_estimators=count, random_state=0, max_depth=best_tree_depth, learning_rate=0.01)
    xgboost_model.fit(X_train_tree, y_train_xg)
    y_pred_xb = xgboost_model.predict(X_test_tree)
    accuracy = accuracy_score(y_test_xg, y_pred_xb)
    f1_xgboost = f1_score(y_test_xg, y_pred_xb, average='macro')
    print(f'XGBoost')
    print(f'The Number of Decision Trees: {count}')
    print(f'Accuracy: {accuracy: .3f}')
    print('---------------------------')
    xgboost_accuracy.append(accuracy)
    if accuracy >= best_accuracy_xb:
        best_accuracy_xb = accuracy
        best_count_xgboost = count
        model_scores['XGBoost'] = f1_xgboost
print(f'Best Number of Decision Trees: {best_count_xgboost}')
print(f'Best Accuracy: {best_accuracy_xb: .3f}')
print('---------------------------')

# A3Q5 Gradient Boosting
tree_count = list(range(80, 91, 10))
gb_accuracy = []
best_accuracy_gb = 0

for count in tree_count:
    gb_model = GradientBoostingClassifier(n_estimators=count, random_state=0)
    gb_model.fit(X_train_tree, y_train_tree)
    y_pred_gb = gb_model.predict(X_test_tree)
    accuracy = accuracy_score(y_test_tree, y_pred_gb)
    f1_gradient = f1_score(y_test_tree, y_pred_gb, average='macro')
    print(f'Gradient Boosting')
    print(f'The Number of Decision Trees: {count}')
    print(f'Accuracy: {accuracy: .3f}')
    print('---------------------------')
    gb_accuracy.append(accuracy)
    if accuracy >= best_accuracy_gb:
        best_accuracy_gb = accuracy
        best_count = count
        model_scores['Gradient Boosting'] = f1_gradient
print(f'Best Number of Decision Trees: {best_count}')
print(f'Best Accuracy: {best_accuracy_gb: .3f}')
print('---------------------------')

# A3Q6 Adam
X_train, X_test, y_train, y_test = train_test_split(X, y_dt, test_size=0.4, random_state=10)
mlp_model = MLPClassifier(solver="adam", random_state=42, max_iter=10000)
mlp_model.fit(X_train, y_train)
y_pred_adam = mlp_model.predict(X_test)
adam_accuracy = accuracy_score(y_test, y_pred_adam)
f1_adam = f1_score(y_test, y_pred_adam, average='macro')
model_scores['adam'] = f1_adam

mlp_model = MLPClassifier(solver="sgd", random_state=42, max_iter=10000)
mlp_model.fit(X_train, y_train)
y_pred_sgd = mlp_model.predict(X_test)
sgd_accuracy = accuracy_score(y_test, y_pred_sgd)
f1_sgd = f1_score(y_test, y_pred_sgd, average='macro')
model_scores['SGD'] = f1_sgd

# Comparison
experiment = ['1', '2']
plt.figure(figsize=(10, 6))
plt.plot(experiment, random_forest_accuracy, marker='o', linestyle='-', color = 'r', label = 'Random Forests')
plt.plot(experiment, xgboost_accuracy, marker='o', linestyle='-', color = 'b', label = 'XGBoost')
plt.plot(experiment, gb_accuracy, marker='o', linestyle='-', color = 'g', label = 'Gradient Boosting')
plt.axhline(y=adam_accuracy , color='c', linestyle='--', label='MLP (Adam)')
plt.axhline(y=sgd_accuracy, color='m', linestyle='--', label='MLP (SGD)')
plt.xlabel('Number of Trees in the Ensemble')
plt.ylabel('Accuracy Score')
plt.title('Comparison of Random Forest, Gradient Boosting, XGBoost, and MLP')
plt.legend()
plt.savefig('Total Comparison.png')
plt.show()

# # A3Q7 weight decay,dropouts, 3 different combinations
# # NN Method
# # Data Processing for NN
# X_tensor = torch.tensor(X.values, dtype=torch.float32)
# y_tensor = torch.tensor((y_log - 1).values)
# X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.4, random_state=10)
#
#
# class LinearNN(nn.Module):
#     def __init__(self, input_size,dropout_rate):
#         super(LinearNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 8)
#         self.dropout = nn.Dropout(dropout_rate)  # 在这里增加一个 dropout 层
#         self.fc3 = nn.Linear(8, 4)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.relu(out)
#         out = self.fc3(out)
#         out = self.softmax(out)
#         return out
#
#
# # Train and Test NN
# def train_and_evaluate(model, X_train, X_test, y_train, y_test, learning_rate,weight_decay,epochs=5000):
#     # Define Loss Function and Optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
#
#     # Training Part
#     for epoch in range(epochs):
#         output = model(X_train)
#         loss = criterion(output, y_train)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     # Testing Part
#     model.eval()
#     with torch.no_grad():
#         y_pred_train = model(X_train).argmax(dim=1)
#         y_pred_test = model(X_test).argmax(dim=1)
#         test_accuracy = accuracy_score(y_pred_test.numpy(), y_test.numpy())
#     return test_accuracy
#
#
# accuracies = []
# input_size = X_train.shape[1]
# hyperparameter_combinations = [
#     {'weight_decay': 0.0001, 'dropout_rate': 0.1, 'learning_rate': 0.001},
#     {'weight_decay': 0.0001, 'dropout_rate': 0.2, 'learning_rate': 0.001},
#     {'weight_decay': 0.001, 'dropout_rate': 0.3, 'learning_rate': 0.005},
# ]
#
# for params in hyperparameter_combinations:
#     model = LinearNN(input_size=input_size, dropout_rate=params['dropout_rate'])
#     test_accuracy = train_and_evaluate(model, X_train, X_test, y_train, y_test, learning_rate=params['learning_rate'],weight_decay=params['weight_decay'])
#     accuracies.append(test_accuracy)
#     print(f"Dropout Rate: {params['dropout_rate']}, Weight Decay: {params['weight_decay']}, Learning Rate: {params['learning_rate']}, Test Accuracy: {test_accuracy:.2f}")

print('---------------------------')
print("Model F1 Scores:", model_scores)
best_models = sorted(model_scores, key=model_scores.get, reverse=True)[:2]
print("Best models:", best_models)
print('---------------------------')

# PartB
# Import Dataset
data = pd.read_csv('data/cmc.data', header=None,
                   names=['Wife_Age', 'Wife_Education', 'Husband_Education', 'Children_Number', 'Wife_Religion',
                          'Wife_Working', 'Husband_Occupation', 'Living_Index', 'Media_Exposure',
                          'Contraceptive_Method'])
data.replace({"?": None, "$": None}, inplace=True)
data.dropna(inplace=True)
X = data.iloc[:, :9]
y = data['Contraceptive_Method']

# Correlation and Heatmap
combined_df = pd.concat([X, y], axis=1)
correlation_matrix = combined_df.corr()
y_correlation = correlation_matrix[['Contraceptive_Method']].drop('Contraceptive_Method')
print('Correlation Map:')
print(y_correlation)
print('---------------------------')
sns.set(style='white')
plt.figure(figsize=(9, 9))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5,
                      xticklabels=X.columns, yticklabels=X.columns)
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
sns.countplot(x=y, alpha=0.7)
plt.title('Histogram of Contraceptive Method Class distribution')
plt.xlabel('Contraceptive_Method')
plt.ylabel('Freq')
plt.savefig('Histogram of Contraceptive Method Class distribution')
plt.show()

labels = ['No-Use', 'Long-Term', 'Short-Term']
plt.figure(figsize=(9, 9))
y.value_counts().plot.pie(autopct='%1.2f%%', startangle=90, cmap='coolwarm', labels=labels)
plt.title('Pie Chart of Contraceptive Method Class distribution')
plt.xlabel('')
plt.ylabel('')
plt.savefig('Pie Chart of Contraceptive Method Class distribution')
plt.show()

# Best Model1: Random Forest
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X, y, test_size=0.4, random_state=10)
best_random_forest_model = RandomForestClassifier(n_estimators=best_count_random_forest, random_state=0)
y_test_bin = label_binarize(y_test_tree, classes=[1, 2, 3])

best_random_forest_model.fit(X_train_tree, y_train_tree)
y_pred_b = best_random_forest_model.predict_proba(X_test_tree)
y_pred = best_random_forest_model.predict(X_test_tree)
accuracy_b = accuracy_score(y_test_tree, y_pred)
f1_b = f1_score(y_test_tree, y_pred, average='macro')
roc_auc_b = roc_auc_score(y_test_bin, y_pred_b, average='macro', multi_class='ovr')


print(f'Best Number of Decision Trees: {best_count}')
print(f'Accuracy: {accuracy_b:.3f}')
print(f'F1 Score: {f1_b:.3f}')
print(f'ROC-AUC: {roc_auc_b:.3f}')
print('---------------------------')

# Best Model2: XGBOOST
y_transformed = y - 1
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.4, random_state=10)
y_test_binarized = label_binarize(y_test + 1, classes=[1, 2, 3])

best_tree_count = best_count_xgboost
xgboost_model = xgb.XGBClassifier(n_estimators=best_tree_count, random_state=0, max_depth=best_tree_depth, learning_rate=0.1)
xgboost_model.fit(X_train, y_train)

y_pred_xgboost = xgboost_model.predict(X_test) + 1
y_prob_xgboost = xgboost_model.predict_proba(X_test)

f1_xgboost = f1_score(y_test + 1, y_pred_xgboost, average='macro')
roc_auc_xgboost = roc_auc_score(y_test_binarized, y_prob_xgboost, multi_class='ovr', average='macro')

print(f'XGBoost with Best Number of Trees ({best_tree_count})')
print(f'F1-Score: {f1_xgboost:.3f}')
print(f'ROC-AUC Score: {roc_auc_xgboost:.3f}')
print('---------------------------')