from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    x = np.load(filename)
    return x - np.mean(x, axis=0)

def get_covariance(dataset):
    return np.dot(np.transpose(dataset), dataset) / (len(dataset)-1)


def get_eig(S, m):
    x = eigh(S, eigvals_only=True)
    y = len(x)
    z = eigh(S, subset_by_index=[y-m,y-1])
    eigval = np.flip(np.diag(z[0]))
    eigvector = np.flip(z[1], axis=1)
    return eigval, eigvector


def get_eig_perc(S, perc):
    x = eigh(S, eigvals_only=True)
    total = sum(x)
    index = 0
    for i in x:
        if(i/total > perc):
            index+=1
    return get_eig(S, index)


def project_image(img, U):
    alpha = np.dot(np.transpose(U), img)
    return np.dot(alpha, np.transpose(U))

def display_image(orig, proj):
    orig = np.transpose(np.reshape(orig, (32,32)))
    proj = np.transpose(np.reshape(proj, (32,32)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,3))
    ax1.set_title('Original')
    ax2.set_title('Projection')
    plt.subplot(ax1)
    plt.imshow(orig, aspect='equal', cmap='viridis')
    plt.colorbar()
    plt.subplot(ax2)
    plt.imshow(proj, aspect='equal', cmap='viridis')
    plt.colorbar()
    plt.show()


x = (get_covariance(np.array([[1,-0.5,-1.5], [-1,0.5,-1.5],[-1,-0.5,1.5],[1,0.5,1.5]])))
print(get_eig(x, 3))