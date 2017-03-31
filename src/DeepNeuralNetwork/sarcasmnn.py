import sys
import tensorflow as tf
import numpy as np
import json
sys.path.append('/Users/FelixDSantos/LeCode/DeepLearning/fyp/src/DataAcquisition')
sys.path.append('/Users/FelixDSantos/LeCode/DeepLearning/fyp/src/utilities')
# print(sys.path)
import create_sarcasm_featuresets as dataprep
import plotutils as utils
# import DataAcquisition.create_sarcasm_featuresets as dataprep
# import utilities.plotutils as utils
import time
import os
import argparse
import pandas as pd

def loadLexAndFeaturesfromjson(path):
    with open(path) as openfile:
            data = json.load(openfile)
            lexicon,features,labels,heldout_x,heldout_y = data[0],np.array(data[1]),np.array(data[2]),np.array(data[3]),np.array(data[4])

            return lexicon,features,labels,heldout_x,heldout_y
# data used in ashwin paper
# sarcasmdataset='/Users/FelixDSantos/LeCode/DeepLearning/fyp/TrainAndTest/sarcasm_set_ashwin.json'
# "Bamman and Smith paper"
# sarcasmdataset ='/Users/FelixDSantos/LeCode/DeepLearning/fyp/TrainAndTest/sarcasm_set_bam_smith.json'
# train_x,train_y,test_x,test_y = loaddatafromjson(sarcasmdataset)

lexicon,features,labels,heldout_x,heldout_y = loadLexAndFeaturesfromjson("/Users/FelixDSantos/LeCode/DeepLearning/fyp/FeatureData/DNNFeatures/bms_Feats_Labels_andHoldout")

lenwholeset=(len(labels)+len(heldout_y))
train_x,train_y,test_x,test_y =dataprep.partitionDataToTrainandTest(features,labels,lenwholeset,80)
test_class=np.argmax(test_y,axis=1)
val_class = np.argmax(heldout_y,axis=1)
helduniq, heldcounts = np.unique(np.argmax(heldout_y,axis=1), return_counts=True)
trainuniq,traincounts = np.unique(np.argmax(train_y,axis=1),return_counts=True)
testuniq,testcounts = np.unique(test_class,return_counts=True)
print("===============================================================")
print("Train/Test/Validation Split : {}/{}/{}".format(len(train_y),len(test_y),len(heldout_y)))
print("Train Data set Non-Sarcastic/Sarcastic Tweets Split: {}/{} ".format(traincounts[0],traincounts[1]))
print("Test Data set Non-Sarcastic/Sarcastic Tweets Split: {}/{} ".format(testcounts[0],testcounts[1]))
print("Held out Data set Non-Sarcastic/Sarcastic Tweets Split: {}/{} ".format(heldcounts[0],heldcounts[1]))
print("===============================================================")
tweet_length= len(train_x[0])
num_neurons1= 100
# num_neurons2=30

num_classes=2
batch_size=300
num_epochs=50
test_every= 50
lamb = 0.005
parser = argparse.ArgumentParser(description='Neural Network Arguments')
parser.add_argument('-summary', action='store', dest='runname',default=str(int((time.time()))),
                    help='Store a runname value')
# parser.add_argument('-model', action='store', dest='modelsavepath',default=str(int((time.time()))),
#                     help='Name of model')
summaryargs = parser.parse_args()
runname=summaryargs.runname
# modelname=summaryargs.modelsavepath

out_dir = os.path.abspath(os.path.join(os.path.curdir,"Neural_Network_Runs",runname))
savepath=os.path.abspath(os.path.join(os.path.curdir,"SavedModels","BmS","DNN_BmS_SavedModel"))
if not os.path.exists(savepath):
    os.makedirs(savepath)
x = tf.placeholder('float',[None, tweet_length], name ="x" )
y=tf.placeholder('float', name= "y")
keep_prob = tf.placeholder("float")

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
    return output,Weights,bias

layer1, Weights_1,bias_1 = create_neural_network_layer(input = x, layer_name="Hidden_Layer_1", num_input = tweet_length , num_outputs=num_neurons1)
l1dropout= tf.nn.dropout(layer1, keep_prob)
# layer2 = create_neural_network_layer(input = layer1, layer_name="Hidden_Layer_2", num_input = num_neurons1 , num_outputs=num_neurons2)
# outputlayer = create_neural_network_layer(input = layer2, layer_name= "Output_Layer", num_input = num_neurons2, num_outputs = num_classes, use_relu=False)
outputlayer,Weights_Out,bias_out = create_neural_network_layer(input = l1dropout, layer_name= "Output_Layer", num_input = num_neurons1, num_outputs = num_classes, use_relu=False)
predictions = tf.argmax(outputlayer, 1, name="Predictions")

with tf.name_scope("Cost"):
    cross_ent = tf.nn.softmax_cross_entropy_with_logits(logits= outputlayer, labels =y)
    cost = tf.reduce_mean(cross_ent)
    # l2 regularization with decaying lambda
    reg = lamb*(tf.nn.l2_loss(Weights_1)+ tf.nn.l2_loss(Weights_Out))
    cost_l2=tf.reduce_mean(cost+ reg)

with tf.name_scope("Optimizer"):
    global_step = tf.Variable(0, name="global_step", trainable= False)
    optimizer = tf.train.AdamOptimizer()
    grads = optimizer.compute_gradients(cost_l2)
    train_model = optimizer.apply_gradients(grads, global_step=global_step)

with tf.name_scope("Accuracy"):
    correct_pred=tf.equal(predictions, tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,'float'), name="Accuracy")

saver = tf.train.Saver()
sess = tf.Session()

"""
Summaries
"""
cost_summary = tf.summary.scalar("Cost", cost_l2)
accuracy_summary = tf.summary.scalar("Accuracy", accuracy)

train_summary_operation = tf.summary.merge([cost_summary,accuracy_summary])
train_summary_loc = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_loc,sess.graph)

test_summary_operation = tf.summary.merge([cost_summary,accuracy_summary])
test_summary_loc = os.path.join(out_dir, "summaries", "test")
test_summary_writer = tf.summary.FileWriter(test_summary_loc,sess.graph)
sess.run(tf.global_variables_initializer())

def train_neural_network(data):
    testconf = np.zeros(shape=(3,3)).astype(int)
    num_batches=0
    for batch in data:
        batch_x, batch_y = zip(*batch)
        # keep 0.5 neurons for training
        feed_dict_train = {x: batch_x, y: batch_y, keep_prob: 0.5}
        _, step,trainsummary,c,trainacc = sess.run([train_model,global_step,train_summary_operation,cost_l2, accuracy],feed_dict=feed_dict_train)
        num_batches +=1
        # print(sess.run(outputlayer,feed_dict=feed_dict_train ))
        print("Training Step: {}|| Cost: {}, Accuracy: {}".format(num_batches,c,trainacc))
        train_summary_writer.add_summary(trainsummary,step)
        if(num_batches%test_every==0):
            step, testsummary,testcost,testacc,testpred = sess.run([global_step,test_summary_operation,cost_l2,accuracy,predictions],{x:test_x,y:test_y,keep_prob:1.0})
            print("===================================Evaluation========================================")
            print("\nEvaluation after {} batches feeded in|| Cost: {}, Accuracy: {}\n".format(num_batches,testcost,testacc))
            print("=====================================================================================")
            test_summary_writer.add_summary(testsummary,step)
            confmatfortest=utils.plot_conf_matrix(test_class,testpred).as_matrix()
            testconf=testconf+(confmatfortest)
    save_path = saver.save(sess, savepath)
    print("Model saved in file: %s" % save_path)
    testconf=pd.DataFrame(testconf, columns=['0','1','All'])
    # testconf.index=['0','1','All']
    # print(testconf)
    testconf.index=['0','1','All']
    print(testconf)
    utils.calculateModelStats(testconf)
        # print('Epoch',epoch+1, 'completed out of', hm_iter,'loss:',epoch_loss)
        # calculate_acc("\nEvaluation: ",test_x,test_y)


# sess= tf.Session()
# sess.run(tf.global_variables_initializer())
# ==========================================
# Uncomment if want to train model
# ==========================================
# beforetrain_pred,beforestep,beforetrain_cost, beforetrain_summary, beforetrain_acc= sess.run([predictions,global_step,cost_l2,test_summary_operation,accuracy],{x:test_x,y:test_y,keep_prob: 1.0})
# print("===================================Evaluation Before Training========================================")
# print("\nCost: {}, Accuracy: {}\n".format(beforetrain_cost,beforetrain_acc))
# print("=====================================================================================================")
# test_summary_writer.add_summary(beforetrain_summary,beforestep)
#
# trainbatches = dataprep.batch_iter(list(zip(train_x, train_y)), batch_size, num_epochs)
# train_neural_network(trainbatches)
#
# test_pred,step,testcost,testsummary,testacc = sess.run([predictions,global_step,cost_l2,test_summary_operation,accuracy],{x:test_x,y:test_y,keep_prob: 1.0})
# print("===================================Evaluation After Training On Testing Set========================================")
# print("\nCost: {}, Accuracy: {}\n".format(testcost,testacc))
# print("===================================================================================================================")
# test_summary_writer.add_summary(testsummary,step)


# Restore variables from disk.
saver.restore(sess, savepath)
print("Model restored.")
# ==========================================
# Uncomment if want to test model on validation set
# ==========================================
val_pred,valcost,valacc=sess.run([predictions,cost,accuracy],{x:heldout_x,y:heldout_y,keep_prob:1.0})
print("")
print("")
print("")
print("===================================Evaluation On Unseen Validation Set of {}========================================".format(len(heldout_y)))
print("\nCost: {}, Accuracy: {}\n".format(valcost,valacc))
print("====================================================================================================================")
validationconfmatrix=utils.plot_conf_matrix(val_class,val_pred)
validationconfmatrix.index=['0','1','All']
validationconfmatrix.columns = ['0', '1','All']
utils.calculateModelStats(validationconfmatrix)
