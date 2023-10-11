import numpy as np
import csv
import math
import random
from matplotlib import pyplot as plt


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    dataset = []
    with open(filename) as csvfile:
        csvfile.readline()
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row[1:])
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset[i][j] = float(dataset[i][j])
    dataset = np.array(dataset)
    return dataset


def print_stats(dataset, col):
    column = dataset[:, col]
    mean = 0
    std = 0
    for i in range(len(column)):
        mean += column[i]
    mean /= len(column)
    for i in range(len(column)):
        std += pow((column[i] - mean), 2)
    std /= len(column) - 1
    std = pow(std, 0.5)
    print('{}\n{:.2f}\n{:.2f}'.format(len(column), mean, std))
    


def regression(dataset, cols, betas):
    mse = 0
    column = []
    temp = 0
    for i in cols:
        column.append(dataset[:, i])
    for i in range(len(column[0])):
        temp = betas[0]
        for j in range(1, len(betas)):
            temp += betas[j] * column[j-1][i]
        temp -= dataset[: , 0][i]
        temp = pow(temp, 2)
        mse += temp
    mse /= len(column[0])
    return mse


def gradient_descent(dataset, cols, betas):
    grads = []
    column = []
    total = 0
    temp = 0
    for i in cols:
        column.append(dataset[:, i]) 
    for i in range(len(column[0])):
        temp = betas[0]
        for j in range(1, len(betas)):
            temp += betas[j] * column[j-1][i]
        temp -= dataset[: , 0][i]
        total += temp
    total *= 2/len(column[0])
    grads.append(total)

    for x in range(1, len(betas)):
        total = 0
        for i in range(len(column[0])):
            temp = betas[0]
            for j in range(1, len(betas)):
                temp += betas[j] * column[j-1][i]
            temp -= dataset[: , 0][i]
            temp *= column[x-1][i]
            total += temp
        total *= 2/len(column[0])
        grads.append(total)
    return np.array(grads)


def iterate_gradient(dataset, cols, betas, T, eta): #order: T, mse, beta0, beta1, beta8
    column = []
    tmse = []
    temp = 0
    prev = betas
    grads = gradient_descent(dataset, cols, betas)
    for i in cols:
        column.append(dataset[:, i])
    for x in range(1, T+1):
        for i in range(len(grads)):
            prev[i] = prev[i] - (eta * grads[i])
        grads = gradient_descent(dataset, cols, prev)
        print('{} {:.2f} '.format(x, regression(dataset, cols, prev)), end='')
        for i in range(len(prev)):
            print('{:.2f} '.format(prev[i]), end = '')
        print('')



def compute_betas(dataset, cols):
    betas = None
    mse = None
    X = []
    y = dataset[:, 0]
    column = []
    for i in cols:
        column.append(dataset[:, i])
    for i in range(len(y)):
        X.append([1])
        for j in range(len(column)):
            X[i].append(column[j][i])
    betas = np.matmul(np.linalg.inv(np.matmul(np.transpose(np.array(X)), np.array(X))), np.transpose(np.array(X)))
    betas = np.matmul(betas, np.array(y))
    mse = regression(dataset, cols, betas)
    return (mse, *betas)


def predict(dataset, cols, features):
    
    betas = compute_betas(dataset, cols)
    result = betas[1]
    for i in range(len(features)):
        result += betas[i+2] * features[i]
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    linear = []
    quadratic = []
    for i in range(len(X)):
        ylinear = betas[0]
        yquadratic = alphas[0]
        z = np.random.normal(loc = 0, scale = sigma)
        for j in range(1, len(betas)):
            ylinear += betas[j] * X[i][0]
            yquadratic += alphas[j] * pow(X[i][0], 2)
        ylinear += z
        yquadratic += z
        linear.append([ylinear, X[i][0]])
        quadratic.append([yquadratic, X[i][0]])
    return np.array(linear), np.array(quadratic)


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph
    X = []
    for i in range(1000):
        X.append([random.randint(-100,101)])
    alphas = np.array([1, 1])
    betas = np.array([1, 2])
    sigmas = []
    for i in range(10):
        sigmas.append(pow(10, i-4))
    datasets1 = []
    datasets2 = []
    for i in range(len(sigmas)):
        datasets1.append(synthetic_datasets(betas, alphas, X, sigmas[i])[0])
        datasets2.append(synthetic_datasets(betas, alphas, X, sigmas[i])[1])
    mse1 = []
    mse2 = []
    for i in range(len(datasets1)):
        mse1.append(compute_betas(datasets1[i], cols=[1])[0])
    for i in range(len(datasets2)):
        mse2.append(compute_betas(datasets2[i], cols=[1])[0])
    f = plt.figure()
    plt.xlabel('Sigmas')
    plt.ylabel('MSEs')
    plt.plot(sigmas, mse1, label = 'linear', marker = 'o')
    plt.plot(sigmas, mse2, label = 'quadratic', marker = 'o')
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    f.savefig("mse.pdf")

if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()

