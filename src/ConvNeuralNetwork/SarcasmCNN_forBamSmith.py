import tensorflow as tf
import numpy as np
import json
import sys
sys.path.append('/Users/FelixDSantos/LeCode/DeepLearning/fyp/src/DataAcquisition')
sys.path.append('/Users/FelixDSantos/LeCode/DeepLearning/fyp/src/utilities')
import create_sarcasm_featuresets as dataprep
# import DataAcquisition.create_sarcasm_featuresets as dataprep
import time
from tensorflow.python import debug as tf_debug
import pandas as pd
import utils as utils
import argparse
import os
from utils import printc

holdouttweettext=dataprep.loaddatafromjson('/Users/FelixDSantos/LeCode/DeepLearning/fyp/FeatureData/Holdout/tweetslabelsAndHoldOut_bms')[2]
data=dataprep.loaddatafromjson('/Users/FelixDSantos/LeCode/DeepLearning/fyp/FeatureData/CNNFeatures/bmsFeats_labelsAndholdout')
vocabsize,tweets,labels,heldout_tweets,heldout_labels=data[0],np.array(data[1]),np.array(data[2]),np.array(data[3]),np.array(data[4])
lenwholeset=(len(labels)+len(heldout_labels))
train_x,train_y,test_x,test_y =dataprep.partitionDataToTrainandTest(tweets,labels,lenwholeset,80)
test_class=np.argmax(test_y,axis=1)
val_class = np.argmax(heldout_labels,axis=1)
helduniq, heldcounts = np.unique(val_class, return_counts=True)
trainuniq,traincounts = np.unique(np.argmax(train_y,axis=1),return_counts=True)
testuniq,testcounts = np.unique(test_class,return_counts=True)
print("===============================================================")
print("Train/Test/Validation Split : {}/{}/{}".format(len(train_y),len(test_y),len(heldout_labels)))
print("Train Data set Non-Sarcastic/Sarcastic Tweets Split: {}/{} ".format(traincounts[0],traincounts[1]))
print("Test Data set Non-Sarcastic/Sarcastic Tweets Split: {}/{} ".format(testcounts[0],testcounts[1]))
print("Held out Data set Non-Sarcastic/Sarcastic Tweets Split: {}/{} ".format(heldcounts[0],heldcounts[1]))
print("===============================================================")
parser = argparse.ArgumentParser(description='Neural Network Arguments')
parser.add_argument('-summary', action='store', dest='runname',default=str(int((time.time()))),
                    help='Store a runname value')
summaryargs = parser.parse_args()
runname=summaryargs.runname

out_dir = os.path.abspath(os.path.join(os.path.curdir,"Neural_Network_Runs",runname))
savepath=os.path.abspath(os.path.join(os.path.curdir,"SavedModels","BmS","CNN_bMs_SavedModel"))
if not os.path.exists(savepath):
    os.makedirs(savepath)
# ================================
#Paramaters
# tweet_length= 376
# num_epochs=20
# batch_size =50
# test_every=15
# data_length = train_x.shape[1]
# num_channels = 1
# num_classes= 2
# filter_sizes=[3,4,5]
# tweet_height=1
# num_filters1=64
# # num_filters2=128
# fc_size= 32
# embedding_size=128
# lamb =0.675
# keepdrop=0.35
# minclip=-0.00000005
# maxclip=0.00000005
num_epochs=23
batch_size =300
test_every=15
data_length = train_x.shape[1]
num_channels = 1
num_classes= 2
filter_sizes=[3,4,5]
tweet_height=1
num_filters1=64
# num_filters2=128
fc_size= 32
embedding_size=128
lamb =0.7
keepdrop=0.475


# ================================
# ================================
#PLACEHOLDERS
x = tf.placeholder(tf.int32, [None,data_length],name="x")
# xreshaped = tf.reshape(x,[-1, data_length,1,num_channels], name="x_tweet")
y = tf.placeholder(tf.float32, [None,num_classes],name="y")
dropoutprob = tf.placeholder(tf.float32, name="dropoutprob")
# dropoutprob = tf.placeholder(tf.float32, name="dropoutprob")
# ================================
# ================================
#Helper methods for creating weights and biases
def create_weights(shape):
    W=tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="W")
    return(W)


def create_bias(length):
    b = tf.Variable (tf.constant(0.1, shape=[length]), name = "b")
    return(b)


def create_conv_layer(input, num_channels,filter_sizes, num_filters,flatten=False):

    pooled_outputs=[]
    for i, filtersize in enumerate(filter_sizes):
        with tf.name_scope("conv-layer-%s" % filtersize):
            filtershape = [filtersize,num_channels, 1,num_filters]

            weights=create_weights(filtershape)
            # tf.Print(weights,[weights], message = "THESE ARE WEIGHTS: ",summarize=2)
            biases=create_bias(length=num_filters)

            convlayer = tf.nn.conv2d(input=input, filter = weights, strides=[1,1,1,1], padding="VALID",name = "conv")

            #Use Rectified linear unit to add a non linearity to our results
            convlayer = tf.nn.bias_add(convlayer, biases)
            reluresult = tf.nn.relu(convlayer, name="relu")

            #we then maxpool our results tweet_length - filtersize+1
            maxpooled= tf.nn.max_pool(reluresult, ksize =[1, data_length - filtersize+1,1,1], strides=[1,1,1,1],padding ='VALID', name="maxpool")
            pooled_outputs.append(maxpooled)
    num_filters_total = num_filters * len(filter_sizes)
    convlayerpooled = tf.concat(len(filter_sizes), pooled_outputs)

    if(flatten==True):
        convlayerpooled = tf.reshape(convlayerpooled, [-1,num_filters_total])
        shape = convlayerpooled.get_shape()
        numfeatures=shape[1:2].num_elements()
        return convlayerpooled,numfeatures
    # print(numfeatures)
    return convlayerpooled

def create_fully_connected_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = create_weights(shape=[num_inputs,num_outputs])
    bias = create_bias(length=num_outputs)
    Output=tf.nn.xw_plus_b(input, weights, bias, name = "Output")
    shape=Output.get_shape()
    # numfeatures=shape[1:2].num_elements()
    # print(numfeatures)
    if use_relu:
        Output= tf.nn.relu(Output)
    return Output,weights,bias


l2_loss = tf.constant(0.0)
# Embedding
with tf.device('/cpu:0'),tf.name_scope("Word_Embeddings"):
    embeddings = tf.Variable(tf.random_uniform([vocabsize, embedding_size], -1.0, 1.0))
    # looks up the vector for each of the source words in the batch
    embedded_chars = tf.nn.embedding_lookup(embeddings, x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars,-1)
convlayer1,num_feat1 = create_conv_layer(input=embedded_chars_expanded, num_channels=embedding_size,filter_sizes=filter_sizes, num_filters=num_filters1, flatten=True)
# convlayer2, num_feat2 = create_conv_layer(input=convlayer1,num_channels= num_filters1, filter_sizes =filter_sizes , num_filters= num_filters2, flatten=True,use_pooling=True)
dropoutlayer1=tf.nn.dropout(convlayer1,dropoutprob)
fclayer1,Weights_1,bias1= create_fully_connected_layer(input=dropoutlayer1, num_inputs=num_feat1, num_outputs=fc_size, use_relu=True)
dropoutlayer2=tf.nn.dropout(fclayer1, dropoutprob)
# print(dropoutlayer.get_shape())
fclayer2,Weights_Out,biasout= create_fully_connected_layer(input=dropoutlayer2, num_inputs=fc_size, num_outputs=num_classes,use_relu=False)
predictions = tf.argmax(fclayer2,1,name="Predictions")

# print(tf.trainable_variables())
with tf.name_scope("Cost"):
    cross_ent = tf.nn.softmax_cross_entropy_with_logits(logits = fclayer2, labels = y)
    # cost = tf.reduce_mean(cross_ent)
    l2_loss+= tf.nn.l2_loss(Weights_1)
    # l2_loss+= tf.nn.l2_loss(bias1)
    l2_loss+= tf.nn.l2_loss(Weights_Out)
    # l2_loss+= tf.nn.l2_loss(biasout)
    # reg = lamb*(tf.nn.l2_loss(Weights_1)+ tf.nn.l2_loss(Weights_Out))
    reg=lamb*l2_loss
    cost_l2=tf.reduce_mean(cross_ent)+reg

with tf.name_scope("Optimizer"):
    global_step = tf.Variable(0, name="global_step", trainable= False)
    optimizer = tf.train.AdamOptimizer()
    grads = optimizer.compute_gradients(cost_l2)
    train_model=optimizer.apply_gradients(grads,global_step=global_step)
    # gradient clipping
    # capped_grads_and_vars = [(tf.clip_by_value(grad, minclip, maxclip), var) for grad, var in grads]
    # train_model=optimizer.apply_gradients(capped_grads_and_vars,global_step=global_step)


    # capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm=0.0005, axes=[0]), gv[1]) for gv in grads]
    # tvars=tf.trainable_variables()
    # grads,_=tf.clip_by_global_norm(tf.gradients(cost_l2,tvars),0.0001)
    # # train_model = optimizer.apply_gradients(capped_grads_and_vars, global_step=global_step)
    # train_model=optimizer.apply_gradients(zip(grads,tvars),global_step=global_step)

with tf.name_scope("Accuracy"):
    correct_pred = tf.equal(predictions, tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,"float"),name="Accuracy")

saver = tf.train.Saver()
session = tf.Session()

"""
Summaries
"""
cost_summary = tf.summary.scalar("Cost", cost_l2)
accuracy_summary = tf.summary.scalar("Accuracy", accuracy)

train_summary_operation = tf.summary.merge([cost_summary,accuracy_summary])
train_summary_loc = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_loc,session.graph)

test_summary_operation = tf.summary.merge([cost_summary,accuracy_summary])
test_summary_loc = os.path.join(out_dir, "summaries", "test")
test_summary_writer = tf.summary.FileWriter(test_summary_loc,session.graph)

session.run(tf.global_variables_initializer())

def train_network(data):
    testconf = np.zeros(shape=(3,3)).astype(int)
    num_batches=0
    for batch in data:
        batch_x, batch_y = zip(*batch)
        # keep 0.5 neurons for training
        feed_dict_train = {x: batch_x, y: batch_y,dropoutprob:keepdrop}
        _, step,trainsummary,c,trainacc = session.run([train_model,global_step,train_summary_operation,cost_l2, accuracy],feed_dict=feed_dict_train)
        num_batches +=1
        print("Training Step: {}|| Cost: {}, Accuracy: {}".format(num_batches,c,trainacc))
        train_summary_writer.add_summary(trainsummary,step)
        if(num_batches%test_every==0):
            step, testsummary,testcost,testacc,testpred = session.run([global_step,test_summary_operation,cost_l2,accuracy,predictions],{x:test_x,y:test_y,dropoutprob:1.0})
            print("===================================Evaluation========================================")
            print("\nEvaluation after {} batches feeded in|| Cost: {}, Accuracy: {}\n".format(num_batches,testcost,testacc))
            print("=====================================================================================")
            test_summary_writer.add_summary(testsummary,step)
            confmatfortest=utils.plot_conf_matrix(test_class,testpred)
            confmatfortest=confmatfortest.as_matrix()
            testconf=np.add(testconf,confmatfortest)
    save_path = saver.save(session, savepath)
    print("Model saved in file: %s" % save_path)
    testconf=pd.DataFrame(testconf, columns=['0','1','All'])
    testconf.index=['0','1','All']
    print(testconf)

    utils.calculateModelStats(testconf)

# ==========================================
# Uncomment if want to train model
# ==========================================
# beforetrain_pred,beforestep,beforetrain_cost, beforetrain_summary, beforetrain_acc= session.run([predictions,global_step,cost_l2,test_summary_operation,accuracy],{x:test_x,y:test_y,dropoutprob:1.0})
# print("===================================Evaluation Before Training========================================")
# print("\nCost: {}, Accuracy: {}\n".format(beforetrain_cost,beforetrain_acc))
# print("=====================================================================================================")
# test_summary_writer.add_summary(beforetrain_summary,beforestep)
#
# trainbatches = dataprep.batch_iter(list(zip(train_x, train_y)), batch_size, num_epochs)
# train_network(trainbatches)


# ==========================================
# Uncomment if want to test model on validation set
# ==========================================


saver.restore(session, savepath)
printc("Model restored.","blue")
# Uncomment if evaluating on held out dataset
val_pred,valcost,valacc=session.run([predictions,cost_l2,accuracy],{x:heldout_tweets,y:heldout_labels,dropoutprob:1.0})
print("")
print("")
print("")
printc("===================================Evaluation On Unseen Validation Set of {}========================================".format(len(heldout_labels)),attributes=['bold'])
print("\nCost: {}, Accuracy: {}\n".format(valcost,valacc))
printc("====================================================================================================================",attributes=['bold'])
validationconfmatrix=utils.plot_conf_matrix(val_class,val_pred)
validationconfmatrix.index=['0','1','All']
validationconfmatrix.columns = ['0', '1','All']
utils.calculateModelStats(validationconfmatrix)
print("\n")
printc("======================================================================================================================","red")
printc("===================================Incorrectly Classified Examples====================================================","red")
utils.showclassifiedexamples(holdouttweettext,val_class,val_pred)
printc("======================================================================================================================","green")
printc("=====================================Correctly Classified Examples====================================================","green")
utils.showclassifiedexamples(holdouttweettext,val_class,val_pred,num=30,correct=True)
