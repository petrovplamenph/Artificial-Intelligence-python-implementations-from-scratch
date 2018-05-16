import pandas as pd
from sklearn.model_selection import train_test_split



mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)
X_mush = mush_df2.iloc[:,1:]
y_mush = mush_df2.iloc[:,[0,1]]
X_train, X_test, y_train, y_test = train_test_split(X_mush, y_mush, random_state=0)


import tensorflow as tf
import numpy as np

train_x = X_train.values.tolist()
train_y =  y_train.values.tolist()
test_x = X_test.values.tolist()
test_y =  y_test.values.tolist()
n_nodes_hl1 = 117
n_nodes_hl2 = 117
n_nodes_hl3 = 117

n_classes = 2
batch_size = 100
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes]))}


# Nothing changes
def neural_network_model(data):

    l1 = tf.matmul(data,hidden_1_layer['weight'])
    l1 = tf.nn.relu(l1)

    l2 = tf.matmul(l1,hidden_2_layer['weight'])
    l2 = tf.nn.relu(l2)

    l3 = tf.matmul(l2,hidden_3_layer['weight'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weight']) 

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
 
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
	    
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])

                batch_y = np.array(train_y[start:end]).reshape(len(train_y[start:end]),2)

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
				                                              y: batch_y})
                epoch_loss += c
                i+=batch_size
				
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

	    
train_neural_network(x)     
