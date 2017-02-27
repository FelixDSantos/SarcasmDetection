import sys
import tensorflow as tf
import numpy as np
import json
from sklearn.metrics import confusion_matrix
sys.path.append('/Users/FelixDSantos/LeCode/DeepLearning/fyp/src/DataAcquisition')
sys.path.append('/Users/FelixDSantos/LeCode/DeepLearning/fyp/src/utilities')
# print(sys.path)
import create_sarcasm_featuresets as dataprep
import plotutils as util
# import DataAcquisition.create_sarcasm_featuresets as dataprep
# import utilities.plotutils as util
import time
import os

#TODO Add l2-regularization

def loaddatafromjson(path):
    with open(path) as openfile:
            data = json.load(openfile)
            return data


# data used in ashwin paper
sarcasmdataset='/Users/FelixDSantos/LeCode/DeepLearning/fyp/TrainAndTest/sarcasm_set_ashwin.json'
# "Bamman and Smith paper"
# sarcasmdataset ='/Users/FelixDSantos/LeCode/DeepLearning/fyp/TrainAndTest/sarcasm_set_bam_smith.json'
train_x,train_y,test_x,test_y = loaddatafromjson(sarcasmdataset)
test_class=np.argmax(test_y,axis=1)

print("Train/Test Split : {}/{}".format(len(train_y),len(test_y)))
tweet_length= len(train_x[0])
num_neurons1= 2000
num_neurons2=1000

num_classes=2
batch_size=300
num_epochs=50
test_every= 100
if(len(sys.argv)==0):
    runname=str(int(time.time()))
else:
    runname=str(sys.argv[1])
out_dir = os.path.abspath(os.path.join(os.path.curdir,"Neural_Network_Runs",runname))

x = tf.placeholder('float',[None, tweet_length], name ="x" )
y=tf.placeholder('float', name= "y")


def create_weights(shape):
    W=tf.Variable(tf.truncated_normal(shape, stddev = 0.1), name="W")
    # W=tf.Variable(tf.random_normal(shape), name="W")
    return W

def create_bias(length):
    b = tf.Variable (tf.constant(0.1, shape=[length]), name = "b")
    # b = tf.Variable(tf.random_normal([length]))
    return(b)

def create_neural_network_layer(input, layer_name, num_input, num_outputs, use_relu = True):
    with tf.name_scope(layer_name):
        shape=[num_input,num_outputs]
        Weights = create_weights(shape)
        bias = create_bias(num_outputs)
        output = tf.add(tf.matmul(input,Weights),bias)
        if use_relu:
            output = tf.nn.relu(output)
    return output

layer1 = create_neural_network_layer(input = x, layer_name="Hidden_Layer_1", num_input = tweet_length , num_outputs=num_neurons1)
layer2 = create_neural_network_layer(input = layer1, layer_name="Hidden_Layer_2", num_input = num_neurons1 , num_outputs=num_neurons2)
outputlayer = create_neural_network_layer(input = layer2, layer_name= "Output_Layer", num_input = num_neurons2, num_outputs = num_classes, use_relu=False)

predictions = tf.argmax(outputlayer, 1, name="Predictions")

with tf.name_scope("Cost"):
    cross_ent = tf.nn.softmax_cross_entropy_with_logits(logits= outputlayer, labels =y)
    cost = tf.reduce_mean(cross_ent)

with tf.name_scope("Optimizer"):
    global_step = tf.Variable(0, name="global_step", trainable= False)
    optimizer = tf.train.AdamOptimizer()
    grads = optimizer.compute_gradients(cost)
    train_model = optimizer.apply_gradients(grads, global_step=global_step)

with tf.name_scope("Accuracy"):
    correct_pred=tf.equal(predictions, tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,'float'), name="Accuracy")

sess = tf.Session()


"""
Summaries
"""
cost_summary = tf.summary.scalar("Cost", cost)
accuracy_summary = tf.summary.scalar("Accuracy", accuracy)

train_summary_operation = tf.summary.merge([cost_summary,accuracy_summary])
train_summary_loc = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_loc,sess.graph)

test_summary_operation = tf.summary.merge([cost_summary,accuracy_summary])
test_summary_loc = os.path.join(out_dir, "summaries", "test")
test_summary_writer = tf.summary.FileWriter(test_summary_loc,sess.graph)

def train_neural_network(x,hm_iter):
    num_batches=0
    for epoch in range(hm_iter):
        epoch_loss=0

        i=0

        while i<len(train_x):
            # take batches of the train
            start=i
            end=i+batch_size

            batch_x=np.array(train_x[start:end])
            batch_y=np.array(train_y[start:end])

            feed_dict_train = {x: batch_x, y: batch_y}
            # optimizing the cost, by modifying the weights
            _, step,trainsummary,c,trainacc = sess.run([train_model,global_step,train_summary_operation,cost, accuracy],feed_dict=feed_dict_train)
            epoch_loss+= c
            i+=batch_size
            num_batches +=1
            print("Training Step: {}|| Cost: {}, Accuracy: {}".format(num_batches,c,trainacc))
            train_summary_writer.add_summary(trainsummary,step)
            if(num_batches%test_every==0):
                step, testsummary,testcost,testacc = sess.run([global_step,test_summary_operation,cost,accuracy],{x:test_x,y:test_y})
                print("===================================Evaluation========================================")
                print("\nEvaluation after {} batches feeded in|| Cost: {}, Accuracy: {}\n".format(num_batches,testcost,testacc))
                print("=====================================================================================")
                test_summary_writer.add_summary(testsummary,step)

        print('Epoch',epoch+1, 'completed out of', hm_iter,'loss:',epoch_loss)
        # calculate_acc("\nEvaluation: ",test_x,test_y)

sess.run(tf.global_variables_initializer())

train_neural_network(x, num_epochs)

test_pred,step,testcost,testsummary,testacc = sess.run([predictions,global_step,cost,test_summary_operation,accuracy],{x:test_x,y:test_y})
print("===================================Evaluation After Training========================================")
print("\nCost: {}, Accuracy: {}\n".format(testcost,testacc))
print("====================================================================================================")
test_summary_writer.add_summary(testsummary,step)


util.plot_conf_matrix(test_class, test_pred)


sess.close()
