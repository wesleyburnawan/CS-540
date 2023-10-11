import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set = datasets.MNIST('./data', train=True, download=True, transform=custom_transform)
    test_set = datasets.MNIST('./data', train=False, transform=custom_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=False)

    if(training == True):
        return train_loader
    else:
        return test_loader

    



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(start_dim=1, end_dim=3),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        print('Train Epoch: {}'.format(epoch) + '   Accuracy: {}/{} ({:.2f}%)'.format(correct, total, correct/total*100)
           +  '   Loss: {:.3f}'.format(running_loss/len(train_loader)))
    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    final_accuracy = 0
    final_loss = 0
    for epoch in range(len(test_loader)):
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data
                opt.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                #loss.backward()
                opt.step()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
        print('Train Epoch: {}'.format(epoch) + '   Accuracy: {}/{} ({:.2f}%)'.format(correct, total, correct/total*100)
           +  '   Loss: {:.3f}'.format(running_loss/len(train_loader)))
        final_loss += running_loss/len(test_loader)
        final_accuracy += total/correct * 100
    if(show_loss==True):
        print('Average loss: {:.3f}'.format(final_loss/len(test_loader)))
    print('Accuracy: {:.2f}%'.format(final_accuracy/len(test_loader)))
    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    prob = F.softmax(logits, dim=1)

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()

train_loader = get_data_loader()
test_loader = get_data_loader(False)
model = build_model()
model.train()
criterion = nn.CrossEntropyLoss()
train_model(model, train_loader, criterion, T = 5)
model.eval()
evaluate_model(model, test_loader, criterion, show_loss = True)

