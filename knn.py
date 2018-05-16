import numpy as np
import random
import csv
from math import sqrt
mapping ={}
reverse_map = {}

def accuracy(y1,y2):
    summs = 0
    for idx in range(len(y1)):
        if y1[idx] == y2[idx]:
            summs += 1
    return summs/len(y1)

def split_data(data_to_split):
    X =[]
    y = []
    for idx in range(len(data_to_split)):
        X.append(data_to_split[idx][:-1])
        y.append(data_to_split[idx][-1])
    return (X, y)
def distance(vect1, vect2):
    summ = 0
    for idx in range(len(vect1)):
        summ += ((vect1[idx] - vect2[idx])**2)
    return  sqrt(summ)
        
def knn(X_train, y_train, X_test,k=5):
    predictions = []
    for idx_test in range(len(X_test)):
        map_dist_to_y ={}
        dist = []
        for idx_train in range(len(X_train)):        
            dis = distance(X_test[idx_test], X_train[idx_train])
            dist.append(dis)
            map_dist_to_y[dis] = y_train[idx_train]
            
        best_k = sorted(dist)[:k]
        votes = {}
        for d in best_k:
            vote = map_dist_to_y[d]
            if vote not in votes:
                votes[vote] =  1
            else:
                votes[vote] = (votes[vote]+1)
        pred = max(votes, key=votes.get)
        predictions.append(pred)
    return predictions
        

def open_f(file='Iris.csv'):
    map_value = -1
    data = []
    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:
            if map_value == -1:
                map_value += 1
                labels = row
                continue
            new_row = []
            for idx in range(len(row)-1):
                new_row.append(float(row[idx]))
            if row[-1] not in mapping:
                map_value += 1
                mapping[row[-1]] = map_value
                reverse_map[map_value] = row[-1]
            new_row.append(mapping[row[-1]])
            data.append(new_row)
        return (data, labels)


data, labels = open_f()

random.seed(1)
random.shuffle(data)

cut_point = 20

test = data[:cut_point]
train = data[cut_point:]
X_train, y_train = split_data(train)
X_test, y_test = split_data(test)
y_pred = knn(X_train, y_train, X_test)
acc = accuracy(y_pred,y_test)
print(acc)
#####################################
from sklearn import sklearn.naive_bayes
from sklearn import metrics 
knn2 = neighbors.KNeighborsClassifier(n_neighbors=5)
knn2.fit(X_train, y_train)
predicted = knn2.predict(X_test)
print(metrics.accuracy_score(y_test,predicted))

















