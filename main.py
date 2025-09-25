import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.algorithms.bipartite import color
from scipy.ndimage import label
from six import binary_type
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, roc_curve, auc, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from sympy.integrals.manualintegrate import exp_rule
import xgboost as xgb
import joblib
from xgboost.dask import predict

# Import Dataset
data = pd.read_csv('data/abalone.data', header=None, names=['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings'])
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
y_correlation = correlation_matrix[['Rings']].drop('Rings')
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
y_log = pd.cut(y, bins = [0, 7, 10, 15, np.inf], labels = [1, 2, 3, 4]).astype(int)
plt.figure(figsize=(9, 9))
sns.countplot(x = y_log, alpha=0.7)
plt.title('Histogram of Class Distribution')
plt.xlabel('Class')
plt.ylabel('Freq')
plt.savefig('Histogram of Class Distribution.png')
plt.show()

# A3Q1 Pie Chart of Class Distribution
labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
plt.figure(figsize=(9, 9))
y_log.value_counts().plot.pie(autopct = '%1.2f%%', startangle = 90, cmap = 'coolwarm', labels = labels)
plt.title('Pie Chart of Class Distribution')
plt.xlabel('')
plt.ylabel('')
plt.savefig('Pie Chart of Class Distribution.png')
plt.show()


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
    y_test_dt =  y_dt.iloc[order[index:]]

    model = DecisionTreeClassifier(random_state = 0, max_depth = tree_depth)
    model.fit(X_train_dt, y_train_dt)
    y_pred_dt = model.predict(X_test_dt)
    accuracy = accuracy_score(y_test_dt, y_pred_dt)
    precision = precision_score(y_test_dt, y_pred_dt, average='macro', zero_division = 0)
    recall = recall_score(y_test_dt, y_pred_dt, average='macro', zero_division = 0)
    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        best_tree_depth = tree_depth
        best_model = model
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
tree.plot_tree(best_model, feature_names = X.columns, class_names = y_dt.astype(str), filled=True)
plt.title("Best Decision Tree Model")
plt.savefig('Best Decision Tree Model.png')
plt.show()


# IF_THEN Rules
rules_dt = export_text(best_model, feature_names = X.columns)
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
    post_pruning_model = DecisionTreeClassifier(random_state = 0, max_depth = best_tree_depth, ccp_alpha = ccp_alpha)
    post_pruning_model.fit(X_train_tree, y_train_tree)
    y_pred_tree = post_pruning_model.predict(X_test_tree)
    accuracy = accuracy_score(y_test_tree, y_pred_tree)
    print(f'Decision Tree Alpha: {ccp_alpha: .3f}')
    print(f'Accuracy: {accuracy: .3f}')
    print('---------------------------')
    post_pruning_accuracy.append(accuracy)
    if accuracy >= best_accuracy_pru:
        best_accuracy_pru = accuracy
        best_alpha = ccp_alpha
        best_model_pru = post_pruning_model

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
tree.plot_tree(best_model_pru, feature_names = X.columns, class_names = y_dt.astype(str), filled=True)
plt.title("Best Decision Tree Model after Post-Pruning")
plt.savefig('Best Decision Tree Model after Post-Pruning.png')
plt.show()


# A3Q4 Random Forests
tree_count = list(range(10, 610, 50))
random_forest_accuracy = []
best_accuracy_rf = 0

for count in tree_count:
    random_tree_model = RandomForestClassifier(n_estimators = count, random_state = 0, max_depth = best_tree_depth, ccp_alpha = best_alpha)
    random_tree_model.fit(X_train_tree, y_train_tree)
    y_pred_rf = random_tree_model.predict(X_test_tree)
    accuracy = accuracy_score(y_test_tree, y_pred_rf)
    print(f'Random Forests')
    print(f'The Number of Decision Trees: {count}')
    print(f'Accuracy: {accuracy: .3f}')
    print('---------------------------')
    random_forest_accuracy.append(accuracy)
    if accuracy >= best_accuracy_rf:
        best_accuracy_rf = accuracy
        best_count = count

print(f'Best Number of Decision Trees: {best_count}')
print(f'Best Accuracy: {best_accuracy_rf: .3f}')
print('---------------------------')

plt.figure(figsize=(9, 9))
plt.plot(tree_count, random_forest_accuracy, marker='o', linestyle='-')
plt.title('Accuracies of Different Numbers of Decision Trees')
plt.xlabel('The Number of Decision Trees')
plt.ylabel('Accuracy')
plt.savefig('Accuracies of Different Numbers of Decision Trees.png')
plt.show()


# A3Q4 XGBoost
xgboost_accuracy = []
best_accuracy_xb = 0
y_train_xg = y_train_tree - 1
y_test_xg = y_test_tree - 1

for count in tree_count:
    xgboost_model = xgb.XGBClassifier(n_estimators = count, random_state = 0, max_depth = best_tree_depth, learning_rate=0.1)
    xgboost_model.fit(X_train_tree, y_train_xg)
    y_pred_xb = xgboost_model.predict(X_test_tree)
    accuracy = accuracy_score(y_test_xg, y_pred_xb)
    print(f'XGBoost')
    print(f'The Number of Decision Trees: {count}')
    print(f'Accuracy: {accuracy: .3f}')
    print('---------------------------')
    xgboost_accuracy.append(accuracy)
    if accuracy >= best_accuracy_xb:
        best_accuracy_xb = accuracy
        best_count = count

print(f'Best Number of Decision Trees: {best_count}')
print(f'Best Accuracy: {best_accuracy_xb: .3f}')
print('---------------------------')


# A3Q4 Gradient Boosting
gb_accuracy = []
best_accuracy_gb = 0

for count in tree_count:
    gb_model = GradientBoostingClassifier(n_estimators = count, random_state = 0, max_depth = best_tree_depth, ccp_alpha = best_alpha)
    gb_model.fit(X_train_tree, y_train_tree)
    y_pred_gb = gb_model.predict(X_test_tree)
    accuracy = accuracy_score(y_test_tree, y_pred_gb)
    print(f'Gradient Boosting')
    print(f'The Number of Decision Trees: {count}')
    print(f'Accuracy: {accuracy: .3f}')
    print('---------------------------')
    gb_accuracy.append(accuracy)
    if accuracy >= best_accuracy_gb:
        best_accuracy_gb = accuracy
        best_count = count

print(f'Best Number of Decision Trees: {best_count}')
print(f'Best Accuracy: {best_accuracy_gb: .3f}')
print('---------------------------')


# A3Q4 Comparison
plt.figure(figsize=(9, 9))
plt.plot(tree_count, random_forest_accuracy, marker='o', linestyle='-', color = 'r', label = 'Random Forests')
plt.plot(tree_count, xgboost_accuracy, marker='o', linestyle='-', color = 'b', label = 'XGBoost')
plt.plot(tree_count, gb_accuracy, marker='o', linestyle='-', color = 'g', label = 'Gradient Boosting')
plt.title('Comparison of Random Forests, XGBoost, and Gradient Boosting')
plt.xlabel('The Number of Decision Trees')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Comparison of Random Forests, XGBoost, and Gradient Boosting.png')
plt.show()




# A3Q5 SGD
# NN Method
# Data Processing for NN
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor((y_log - 1).values)
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.4, random_state=10)


class LinearNN(nn.Module):
    def __init__(self, input_size):
        super(LinearNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 8)
        self.fc3 = nn.Linear(8, 4)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out


# Train and Test NN
def train_and_evaluate(model, X_train, X_test, y_train, y_test, learning_rate, epochs=5000):
    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training Part
    for epoch in range(epochs):
        output = model(X_train)
        loss = criterion(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Testing Part
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train).argmax(dim = 1)
        y_pred_test = model(X_test).argmax(dim = 1)
        train_accuracy = accuracy_score(y_pred_train.numpy(), y_train.numpy())
        test_accuracy = accuracy_score(y_pred_test.numpy(), y_test.numpy())

        print(f'SGD')
        print(f'Train Accuracy: {train_accuracy:.4f}')
        print(f'Test Accuracy: {test_accuracy:.4f}')
        print('---------------------------')

    return test_accuracy



model = LinearNN(input_size=X_train.shape[1])
test_accuracy = train_and_evaluate(model, X_train, X_test, y_train, y_test, learning_rate=0.1)




