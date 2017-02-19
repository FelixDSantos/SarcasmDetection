import sys
import tensorflow as tf
import numpy as np
import json

def loaddatafromjson(path):
    with open(path) as openfile:
            data = json.load(openfile)
            return data


# data used in ashwin paper
sarcasmdataset='/Users/FelixDSantos/LeCode/DeepLearning/fyp/TrainAndTest/sentiment_set_nolemmatize.json'
train_x,train_y,test_x,test_y = loaddatafromjson(sarcasmdataset)



n_nodes_hl1= 2000
n_nodes_hl2=1000

n_classes=2
batch_size=300
# tells the network to go through batches of 100 of features feed through the network, manipulate the weights and do another 100 and so on


x = tf.placeholder('float',[None, len(train_x[0])])
y=tf.placeholder('float')

# (inputdata*weights) +biases
# we have a bias because if all the input data is 0, then 0*weights means we'd get a 0 and so no neuron would fire
# adding a bias would just make it so that at least some neurons would fire, even if the inputs are 0

def neural_network_model(data):
    '''
    computation graph
    this is our tensor flow model/nn model
    '''
    hidden_1_layer={'weights': tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer={'weights': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    # hidden_3_layer={'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer={'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    # goes through a threshold function/activation function {sigmoid} or {rectified linear}
    # l1=tf.sigmoid(l1)
    l1=tf.nn.relu(l1)

    l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    # l2=tf.sigmoid(l2)
    l2=tf.nn.relu(l2)

    # l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    # l3=tf.nn.relu(l3)

    output=tf.matmul(l2,output_layer['weights'])+output_layer['biases']

    return output

sess=tf.Session()



prediction= neural_network_model(x)
def train_neural_network(x,hm_iter=10):
    # prediction= neural_network_model(x)
    # using cross entropy with logits as the cost function
    # which calculates the difference between the prediction we got vs the known label
    cross_ent = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels = y)
    cost= tf.reduce_mean( cross_ent)
    # we want to minimize this cost

    # learning_rate = 0.001
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    # hm_epochs = 20
    sess.run(tf.global_variables_initializer())
    # with tf.Session() as sess:
        # this begins the session
    # sess.run(tf.initialize_all_variables())

    # train the network here:
    for epoch in range(hm_iter):
        epoch_loss=0
        # for _ in range(int(mnist.train.num_examples/batch_size)):
        #     # chunks through the dataset for you
        #     epoch_x,epoch_y = mnist.train.next_batch(batch_size)
        i=0
        while i<len(train_x):
            # take batches of the train
            start=i
            end=i+batch_size

            batch_x=np.array(train_x[start:end])
            batch_y=np.array(train_y[start:end])

            feed_dict_train = {x: batch_x, y: batch_y}
            # optimizing the cost, by modifying the weights
            _, c = sess.run([optimizer,cost],feed_dict=feed_dict_train)
            epoch_loss+= c
            i+=batch_size
        print('Epoch',epoch+1, 'completed out of', hm_iter,'loss:',epoch_loss)
        # once these weights are trained
        # we compare on the actual label
        # tf.argmax is gonna return the idnex of the maximum value in the array
        # hope that these indexes are the same


        # feed_dict_test = {x: test_x,y:test_y}
        # correct= tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        #
        # accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        # print('Accuracy',accuracy.eval({x:test_x,y:test_y}))

def calculate_acc():
    # prediction=neural_network_model(x)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    acc = sess.run(accuracy,feed_dict={x:test_x,y:test_y})
    print("Accuracy:",acc)

hm_iter=int(sys.argv[1])
train_neural_network(x,hm_iter)
calculate_acc()
# sess.close()
