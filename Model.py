import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import correlate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



def import_data():
    data = pd.read_csv('data/abalone.data', sep = ',', header = None)
    data = data.values
    data[:,0] = np.where(data[:,0] == 'F', 0, np.where(data[:,0] == 'M', 1, 2))
    data[:,8] = data[:,8].astype(float) + 1.5
    return data

def heat_map(name, data):
    sns.set(style = 'white')
    plt.figure()
    heatmap = sns.heatmap(data, annot = True, fmt = ".2f", cmap = 'coolwarm', square= True , linewidths = 0.5, xticklabels = name, yticklabels = name)
    plt.title('Heatmap of abalone figures')
    plt.show()

def scatter_plot(name_x, name_y, data_x, data_y):
    sns.set(style='white')
    plt.figure()
    sns.scatterplot(x = data_x, y = data_y, alpha = 0.6)
    plt.title('Scatter plot of ' +  name_x + ' vs ' + name_y)
    plt.xlabel(name_x)
    plt.ylabel(name_y)
    plt.show()

def hist_plot(name, data):
    plt.figure()
    plt.hist(data)
    plt.title(name + '_hist.png')
    plt.show()
    # plt.savefig(name + '_hist.png', dpi = 500)

def data_model(data):
    np.random.seed(0)
    experiment_number = 1
    random_seed = experiment_number
    x = data[:, :8]
    y = data[:, 8]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = random_seed)





def main():
    abalone = import_data().astype(float)
    correlate_matrix = np.corrcoef(abalone, rowvar = False)
    abalone_x = abalone[:, :8]
    abalone_y = abalone[:, 8]
    figure_name = ['sex','length','diameter','height','whole weight','shucked weight','viscera weight','shell weight','age']

    scatter_plot(figure_name[2], figure_name[8], abalone[:, 2], abalone[:, 8])
    scatter_plot(figure_name[7], figure_name[8], abalone[:, 7], abalone[:, 8])

    heat_map(figure_name, correlate_matrix)

    for i in range(len(abalone_x[0])):
        name = figure_name[i]
        data = abalone_x[:, i]
        hist_plot(name, data)

    data_model(abalone)



if __name__ == '__main__':
    main()