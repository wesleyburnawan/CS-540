import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from random import randint

def load_data(filepath):
    pokemonList = []
    with open(filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if(len(pokemonList) < 20):
                pokemonList.append(row)
    for i in range(len(pokemonList)):
        pokemonList[i]['#'] = int(pokemonList[i]['#'])
        pokemonList[i]['Total'] = int(pokemonList[i]['Total'])
        pokemonList[i]['HP'] = int(pokemonList[i]['HP'])
        pokemonList[i]['Attack'] = int(pokemonList[i]['Attack'])
        pokemonList[i]['Defense'] = int(pokemonList[i]['Defense'])
        pokemonList[i]['Sp. Atk'] = int(pokemonList[i]['Sp. Atk'])
        pokemonList[i]['Sp. Def'] = int(pokemonList[i]['Sp. Def'])
        pokemonList[i]['Speed'] = int(pokemonList[i]['Speed'])
        pokemonList[i].pop('Generation')
        pokemonList[i].pop('Legendary')
    return pokemonList

def calculate_x_y(stats):
    return (stats['Attack'] + stats['Sp. Atk'] + stats['Speed'], stats['HP'] +
      stats['Defense'] + stats['Sp. Def'])

def hac(dataset):
    for i in reversed(dataset):
        if(not math.isfinite(i[0]) or not math.isfinite(i[1])):
            dataset.remove((i[0], i[1]))
    data = [[None for _ in range(4)] for _ in range(len(dataset)-1)]
    clusters = []
    for i in range(len(dataset)):
        clusters.append([i, None, [i]])
    rank = []
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            if(i < j):
                rank.append([i, j, math.dist(dataset[i], dataset[j])])
    rank = sorted(rank, key = lambda l:l[2])
    rankIndex = 0

    for i in range(len(data)):
        if(clusters[rank[rankIndex][0]][1] == None and clusters[rank[rankIndex][1]][1] == None):
            data[i][0] = rank[rankIndex][0]
            data[i][1] = rank[rankIndex][1]
            data[i][2] = rank[rankIndex][2]
            clusters.append([len(clusters), None,
            (clusters[rank[rankIndex][0]][2] + clusters[rank[rankIndex][1]][2])])
            clusters[rank[rankIndex][0]][1] = len(clusters)-1
            clusters[rank[rankIndex][1]][1] = len(clusters)-1
            rankIndex += 1

        elif(clusters[rank[rankIndex][0]][1] == None):
            data[i][0] = rank[rankIndex][0]
            index = rank[rankIndex][1]
            while(clusters[index][1] != None):
                index = clusters[index][1]
            data[i][1] = clusters[index][0]
            data[i][2] = rank[rankIndex][2]
            clusters.append([len(clusters), None, 
            (clusters[rank[rankIndex][0]][2] + clusters[index][2])])
            clusters[rank[rankIndex][0]][1] = len(clusters)-1
            clusters[index][1] = len(clusters)-1
            rankIndex += 1

        elif(clusters[rank[rankIndex][1]][1] == None):
            data[i][1] = rank[rankIndex][1]
            index = rank[rankIndex][0]
            while(clusters[index][1] != None):
                index = clusters[index][1]
            data[i][0] = clusters[index][0]
            data[i][2] = rank[rankIndex][2]
            clusters.append([len(clusters), None,
            (clusters[rank[rankIndex][1]][2] + clusters[index][2])])
            clusters[rank[rankIndex][1]][1] = len(clusters)-1
            clusters[index][1] = len(clusters)-1
            rankIndex += 1

        else:
            index1 = rank[rankIndex][0]
            index2 = rank[rankIndex][1]
            while(clusters[index1][1] != None):
                index1 = clusters[index1][1]
            while(clusters[index2][1] != None):
                index2 = clusters[index2][1]
            while(index1 == index2):
                rankIndex += 1
                index1 = rank[rankIndex][0]
                index2 = rank[rankIndex][1]
                while(clusters[index1][1] != None):
                    index1 = clusters[index1][1]
                while(clusters[index2][1] != None):
                    index2 = clusters[index2][1]
            data[i][0] = clusters[index1][0]
            data[i][1] = clusters[index2][0]
            data[i][2] = rank[rankIndex][2]
            clusters.append([len(clusters), None,
            (clusters[index1][2] + clusters[index2][2])])
            clusters[index1][1] = len(clusters)-1
            clusters[index2][1] = len(clusters)-1
            rankIndex += 1

    #fill up the total points in a cluster
    for i in range(len(data)):
        data[i][3] = len(clusters[data[i][0]][2]) + len(clusters[data[i][1]][2])
    #swap cluster if first cluster is bigger than second
    for i in range(len(data)):
        if(data[i][0] > data[i][1]):
            data[i][0], data[i][1] = data[i][1], data[i][0]
    #address tiebreaker
    for i in range(len(data)-1):
        if(data[i][2] == data[i+1][2]):
            if(data[i][0] > data[i+1][0]):
                data[i], data[i+1] = data[i+1], data[i]
            elif(data[i][0] == data[i+1][0]):
                if(data[i][1] > data[i+1][1]):
                    data[i], data[i+1] = data[i+1], data[i]
    return np.matrix(data)

def random_x_y(m):
    values = []
    for i in range(m):
        values.append((randint(1,359), randint(1,359)))
    return values

def imshow_hac(dataset):
    for i in reversed(dataset):
        if(not math.isfinite(i[0]) or not math.isfinite(i[1])):
            dataset.remove((i[0], i[1]))
    for i in range(len(dataset)):
        plt.scatter(dataset[i][0], dataset[i][1])
    
    data = [[None for _ in range(4)] for _ in range(len(dataset)-1)]
    clusters = []
    for i in range(len(dataset)):
        clusters.append([i, None, [i]])
    rank = []
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            if(i < j):
                rank.append([i, j, math.dist(dataset[i], dataset[j])])
    rank = sorted(rank, key = lambda l:l[2])
    rankIndex = 0

    for i in range(len(data)):
        if(clusters[rank[rankIndex][0]][1] == None and clusters[rank[rankIndex][1]][1] == None):
            data[i][0] = rank[rankIndex][0]
            data[i][1] = rank[rankIndex][1]
            x_values = [dataset[rank[rankIndex][0]][0], dataset[rank[rankIndex][1]][0]]
            y_values = [dataset[rank[rankIndex][0]][1], dataset[rank[rankIndex][1]][1]]
            plt.plot(x_values, y_values)
            plt.pause(0.1)
            data[i][2] = rank[rankIndex][2]
            clusters.append([len(clusters), None,
            (clusters[rank[rankIndex][0]][2] + clusters[rank[rankIndex][1]][2])])
            clusters[rank[rankIndex][0]][1] = len(clusters)-1
            clusters[rank[rankIndex][1]][1] = len(clusters)-1
            rankIndex += 1

        elif(clusters[rank[rankIndex][0]][1] == None):
            
            data[i][0] = rank[rankIndex][0]
            index = rank[rankIndex][1]
            while(clusters[index][1] != None):
                index = clusters[index][1]

            x_values = [dataset[rank[rankIndex][0]][0], dataset[rank[rankIndex][1]][0]]
            y_values = [dataset[rank[rankIndex][0]][1], dataset[rank[rankIndex][1]][1]]
            plt.plot(x_values, y_values)
            plt.pause(0.1)

            data[i][1] = clusters[index][0]
            data[i][2] = rank[rankIndex][2]
            clusters.append([len(clusters), None, 
            (clusters[rank[rankIndex][0]][2] + clusters[index][2])])
            clusters[rank[rankIndex][0]][1] = len(clusters)-1
            clusters[index][1] = len(clusters)-1
            rankIndex += 1

        elif(clusters[rank[rankIndex][1]][1] == None):
            data[i][1] = rank[rankIndex][1]
            index = rank[rankIndex][0]
            while(clusters[index][1] != None):
                index = clusters[index][1]

            x_values = [dataset[rank[rankIndex][0]][0], dataset[rank[rankIndex][1]][0]]
            y_values = [dataset[rank[rankIndex][0]][1], dataset[rank[rankIndex][1]][1]]
            plt.plot(x_values, y_values)
            plt.pause(0.1)

            data[i][0] = clusters[index][0]
            data[i][2] = rank[rankIndex][2]
            clusters.append([len(clusters), None,
            (clusters[rank[rankIndex][1]][2] + clusters[index][2])])
            clusters[rank[rankIndex][1]][1] = len(clusters)-1
            clusters[index][1] = len(clusters)-1
            rankIndex += 1

        else:
            index1 = rank[rankIndex][0]
            index2 = rank[rankIndex][1]
            while(clusters[index1][1] != None):
                index1 = clusters[index1][1]
            while(clusters[index2][1] != None):
                index2 = clusters[index2][1]
            while(index1 == index2):
                rankIndex += 1
                index1 = rank[rankIndex][0]
                index2 = rank[rankIndex][1]
                while(clusters[index1][1] != None):
                    index1 = clusters[index1][1]
                while(clusters[index2][1] != None):
                    index2 = clusters[index2][1]
            
            x_values = [dataset[rank[rankIndex][0]][0], dataset[rank[rankIndex][1]][0]]
            y_values = [dataset[rank[rankIndex][0]][1], dataset[rank[rankIndex][1]][1]]
            plt.plot(x_values, y_values)
            plt.pause(0.1)
            
            data[i][0] = clusters[index1][0]
            data[i][1] = clusters[index2][0]
            data[i][2] = rank[rankIndex][2]
            clusters.append([len(clusters), None,
            (clusters[index1][2] + clusters[index2][2])])
            clusters[index1][1] = len(clusters)-1
            clusters[index2][1] = len(clusters)-1
            rankIndex += 1
        
    plt.show()

test = load_data('Pokemon.csv')

x_y = [calculate_x_y(stats) for stats in load_data('Pokemon.csv')]
imshow_hac(x_y)