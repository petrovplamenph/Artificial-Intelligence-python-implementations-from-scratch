from scipy.io import arff
import pandas as pd
from statistics import mean
import random
import math

"""
#TEST DATA
labels = ['sky','air','hum','wind','watter','forecast','like']
data =[('sun','hot','normal','strong','hot','same',1),
       ('sun','hot','high','strong','hot','same',1),
       ('rain','cold','high','strong','hot','change',0),
       ('sun','hot','high','strong','cold','change',1)]

df = pd.DataFrame.from_records(data,columns=labels)

df.loc[len(df)]=['sun','hot','normal','week','hot','same',0]
"""
data, meta = arff.loadarff('breast-cancer.arff')
cancer_df = pd.DataFrame(data)
dummy_target_column = (cancer_df.iloc[:,-1]==cancer_df.iloc[0,-1])
#first element has value 'recurrence-events',and this value will be defined as the positive class
cancer_df.iloc[:,-1] = dummy_target_column

def calculate_entropy(list_counts):
    all_elements = sum(list_counts)
    entropy = 0
    for element in list_counts:
        if element != 0:
            entropy += -math.log2(element/all_elements)*(element/all_elements)
        
    return entropy


def gain(inital_distrib, distributions_after):
    inital_distrib, distributions_after
    total=sum(inital_distrib)
    gain = calculate_entropy(inital_distrib)
    for split in distributions_after:
        total_after = sum(split)
        k = total_after/total
        split_entropy = calculate_entropy(split)
        gain -= k*split_entropy
    return gain
    

def calculate_gain(df):
    cols = df.columns.values.tolist()
    target = cols[-1]
    ipos = df[df[target] == 1].shape[0]
    ineg = df[df[target] == 0].shape[0]
    initial_distribution = [ipos, ineg]
    max_gain = -1
    max_atribute = ''
    for atribute in cols[:-1]:
        features = df[atribute].unique()
        feature_distribution = []
        col = df[atribute]
        for feature in features:
            pos_feat = col[(col == feature)  &  (df[target] == 1)].shape[0]
            neg_feat = col[(col == feature)  &  (df[target] == 0)].shape[0]
            feature_distribution.append([pos_feat, neg_feat])

        entropy_gain = gain(initial_distribution,feature_distribution)
        if entropy_gain > max_gain:
            max_gain = entropy_gain
            max_atribute = atribute
    return(max_atribute)

class node():
    def __init__(self, children=[], atribute='leafnext',classification = 'A'):
        self.children = children
        self.atribute = atribute
        self.classification = classification
        self.feature_vals_to_node = {}
        self.el_Id = 'node'
        
    def __str__(self):
        return str(self.atribute)
    def getId(self):
        return self.el_Id

    def setAtt(self, val):
        self.atribute = val
        
    def getAtt(self):
        return self.atribute
    def addfeature_vals_to_node(self, feat, next_node):
        self.feature_vals_to_node[feat] = next_node     
        
    def addChild(self, child):
        self.children.append(child)
        
    def getChildren(self):
        dictonary = self.feature_vals_to_node
        return (dictonary,list(dictonary.keys()))
    
    def setClassification(self,val):
        self.classification = val
        
    def getClassification(self):
        return self.classification

class leaf():
    def __init__(self,feature_val='',classification = 'A'):
        self.feature_val = feature_val
        self.classification = classification
        self.el_id = 'leaf'
        
    def getId(self):
        return self.el_id
    
    def setClassification(self,val):
        self.classification = val
        
    def getClassification(self):
        return self.classification
    def getFeature_val(self):
        return self.feature_val


def fit(data, last_node):
    cols = data.columns.values.tolist()
    target = cols[-1]
    number_cols = data.shape[0]
    if data.shape[1] == 1:
        new_leaf = leaf('whatever', data.mean()>0.5)
        last_node.addfeature_vals_to_node('whatever', new_leaf)
        return last_node
    if data[data[target] == 1].shape[0] == number_cols: 
        new_leaf = leaf('whatever', 1)
        last_node.addfeature_vals_to_node('whatever', new_leaf)
        return last_node
    
    if data[data[target] == 0].shape[0] == number_cols: 
        new_leaf = leaf('whatever', 0)        
        last_node.addfeature_vals_to_node('whatever', new_leaf)
        return last_node

    atribute = calculate_gain(data)
 
    features = data[atribute].unique()
    col = data[atribute]
    pos_total = 0
    neg_total = 0
    for feature in features:
        pos_feat = col[(col == feature)  &  (data[target] == 1)].shape[0]
        pos_total += pos_feat
        neg_feat = col[(col == feature)  &  (data[target] == 0)].shape[0]
        neg_total += neg_feat
        if neg_feat == 0:
            
            new_leaf = leaf(feature, 1)
            last_node.addfeature_vals_to_node(feature, new_leaf)
        elif pos_feat == 0:
            new_leaf = leaf(feature, 0)
            last_node.addfeature_vals_to_node(feature, new_leaf)
        else:
            next_data = data[data[atribute] == feature]             
            next_data = next_data.drop([atribute], axis=1)
            new_node = node()
            fit(next_data, new_node)
            last_node.addfeature_vals_to_node(feature, new_node)
    last_node.setAtt(atribute)
    unseen_value_class = int(pos_total>neg_total)
    last_node.setClassification(unseen_value_class)
    return last_node
root = node()            
fit(cancer_df, root)


def preddict_example(example,current_node):
    test_atribute = current_node.getAtt()
    if test_atribute == 'leafnext':
        children, key = current_node.getChildren()
        return children[key[0]].getClassification()
    sample_atribite_val = example[test_atribute].values[0]
    children, keys = current_node.getChildren()
    for i in range(len(keys)):
        feature = keys[i]
        if feature == 'whatever':
            return children[keys[i]].getClassification()
        if feature == sample_atribite_val:
            if children[keys[i]].getId() == 'leaf':
                return children[keys[i]].getClassification()
            else:
                return preddict_example(example,children[keys[i]])
    return current_node.getClassification()
 
def predict(data,three):
    y_vect = []
    for idx in range(len(data)):
        example = data.iloc[[idx],:]
        prdiction = preddict_example(example,three)
        y_vect.append(prdiction)
    return y_vect
def accuracy(y1,y2):
    summs = 0

    for idx in range(len(y1)):
        if int(y1[idx]) == int(y2[idx]):
            summs += 1
    return summs/len(y1)
      
def crossval(data):
    cols = data.columns.values.tolist()
    data = data.values.tolist()
    random.seed(2)
    random.shuffle(data)
    cut_interval = int(len(data)/10)
    avg_acc = []
    for idx in range(10):
        start = idx*cut_interval
        stop = start + cut_interval
        test = data[start:stop]
        test = pd.DataFrame(test, columns=cols)
        train = data[0:start] + data[stop:]
        train = pd.DataFrame(train, columns=cols)
        three = node()
        fit(train, three)               
        y_pred = predict(test,three) 
        y_test = test[cols[-1]].tolist()
        
        acc = accuracy(list(y_pred),list(y_test))
        avg_acc.append(acc)
    mean_acc = mean(avg_acc)
    print('mean',mean_acc)
    return mean_acc
crossval(cancer_df)