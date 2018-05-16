import pandas as pd
from statistics import mean

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
        X.append(data_to_split[idx][1:])
        y.append(data_to_split[idx][0])
    return (X, y)

class NB:
    def __init__(self,X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        LUT = []
        row_names = []
        for name in y_train:
            if name not in row_names:
                row_names.append(name)
        columns = range(len(X_train[0]))
        for i in columns:
            LUT.append([0,0])
        LUT.append(row_names)    
        self.LUT = LUT
        self.proba_f = {}
           
    def fit(self):
        #default_value = 1/len(self.X_train)
        default_value = 0
        for idx in range(len(self.LUT[-1])):
            target_row = self.LUT[-1][idx]
            for column in range(len(self.LUT)-1):
                value = default_value
                values = []
                for index in range(len(self.X_train)):
                    if self.y_train[index] != target_row:
                        continue
                    values.append(self.X_train[index][column])
                value += mean(values)
                self.LUT[column][idx] = value
        for v in self.y_train:
            if v not in self.proba_f:
                self.proba_f[v] =  1
            else:
                self.proba_f[v] = self.proba_f[v]+1
    def predict(self, X_test):
        result = []
        for row in X_test:
            prob = {}
            for key in self.proba_f:
                prob[key] = 1
                LUT_row = self.LUT[-1].index(key)              
                for idx in range(len(row)):
                    if row[idx] == 0.5:
                        continue
                    prob_to_vote_yes = self.LUT[idx][LUT_row]
                    if row[idx] == 1.0:
                        prob[key] = prob[key]*prob_to_vote_yes
                    else:
                        prob[key] = prob[key]*(1-prob_to_vote_yes)
            pred = max(prob, key=prob.get)
            result.append(pred)
        return result
                    
df = pd.read_table('house-votes-84.data.txt', sep=',',header=None)
df.replace(to_replace='n', value=0, inplace=True)
df.replace(to_replace='?', value=0.5, inplace=True)
df.replace(to_replace='y', value=1, inplace=True)        
data = df.values.tolist()
random.seed(5)
random.shuffle(data)
cut_interval = int(len(data)/10)
avg_acc = []
for idx in range(10):
    start = idx*cut_interval
    stop = start + cut_interval
    test = data[start:stop]
    train = data[0:start] + data[stop:]
    
    X_train, y_train = split_data(train)
    X_test, y_test = split_data(test)
    clf = NB(X_train, y_train)
    clf.fit()                 
    y_pred = clf.predict(X_test) 
    acc = accuracy(y_pred,y_test)
    avg_acc.append(acc)
mean_acc = mean(avg_acc)
print(mean_acc)




############################################
from sklearn import naive_bayes
from sklearn import metrics
avg_acc = []
for idx in range(10):
    start = idx*cut_interval
    stop = start + cut_interval
    test = data[start:stop]
    train = data[0:start] + data[stop:]
    
    X_train, y_train = split_data(train)
    X_test, y_test = split_data(test)
    nb = naive_bayes.MultinomialNB()
    nb.fit(X_train, y_train)
    predicted = nb.predict(X_test)
    acc = metrics.accuracy_score(predicted,y_test)
    avg_acc.append(acc)
mean_acc = mean(avg_acc)
print(mean_acc) 

                 
                    
                    
                    