import tensorflow as tf
import numpy as np
import json
import sys
sys.path.append('/Users/FelixDSantos/LeCode/DeepLearning/fyp/src/DataAcquisition')
import create_sarcasm_featuresets as dataprep
# import DataAcquisition.create_sarcasm_featuresets as dataprep
import time
from tensorflow.python import debug as tf_debug
def loaddatafromjson(path):
    with open(path) as openfile:
        data = json.load(openfile)
        return data

sentimentdataset='/Users/FelixDSantos/LeCode/DeepLearning/sentimentdata.json'
sarcasmdataset='/Users/FelixDSantos/LeCode/DeepLearning/fyp/TrainAndTest/sentiment_set_nolemmatize.json'
# train_x,train_y,test_x,test_y = loaddatafromjson(sarcasmdataset)
train_x,train_y,test_x,test_y = loaddatafromjson(sentimentdataset)
# ================================
#Paramaters
# tweet_length= 376
data_length = 423
num_channels = 1
num_classes= 2
filter_sizes=[3,4,5]
tweet_height=1
num_filters1=64
num_filters2=128
fc_size= 64
# ================================
# ================================
#PLACEHOLDERS
x = tf.placeholder(tf.float32, [None,data_length],name="x")
xreshaped = tf.reshape(x,[-1, data_length,1,num_channels], name="x_tweet")
y = tf.placeholder(tf.float32, [None,num_classes],name="y")
dropoutprob = tf.placeholder(tf.float32, name="dropoutprob")
# ================================
# ================================
#Helper methods for creating weights and biases
def create_weights(shape):
    W=tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="W")
    return(W)


def create_bias(length):
    b = tf.Variable (tf.constant(0.1, shape=[length]), name = "b")
    return(b)


def create_conv_layer(input, num_channels,filter_sizes, num_filters,flatten=False, use_pooling=True):

    pooled_outputs=[]
    for i, filtersize in enumerate(filter_sizes):
        with tf.name_scope("conv-layer-%s" % filtersize):
            filtershape = [filtersize,1, 1,num_filters]

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
    return Output

# TODO REMOVE THIS
xbatch1 , ybatch1 = train_x[1:5], train_y[1:5]

convlayer1,num_feat1 = create_conv_layer(input=xreshaped, num_channels=num_channels,filter_sizes=filter_sizes, num_filters=num_filters1, flatten=True,use_pooling=True)
# convlayer2, num_feat2 = create_conv_layer(input=convlayer1,num_channels= num_filters1, filter_sizes =filter_sizes , num_filters= num_filters2, flatten=True,use_pooling=True)
fclayer1= create_fully_connected_layer(input=convlayer1, num_inputs=num_feat1, num_outputs=fc_size)
dropoutlayer=tf.nn.dropout(fclayer1, dropoutprob)
# print(dropoutlayer.get_shape())
fclayer2= create_fully_connected_layer(input=dropoutlayer, num_inputs=fc_size, num_outputs=num_classes,use_relu=False)
predictions = tf.argmax(fclayer2,1,name="Predictions")

with tf.name_scope("Cost"):
    cross_ent = tf.nn.softmax_cross_entropy_with_logits(logits = fclayer2, labels = y)
    cost = tf.reduce_mean(cross_ent)

with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

with tf.name_scope("Accuracy"):
    correct_pred = tf.equal(predictions, tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,"float"),name="Accuracy")


def train_network(data):
    i=0
     #start-time used for printing time-usage
    start_time = time.time()
    for batch in data:
        xbatch, ybatch = zip(*batch)
        feed_dict = {x:xbatch , y:ybatch, dropoutprob:0.5}

        _, acc,loss=session.run([optimizer,accuracy,cost], feed_dict=feed_dict)

        i+=1
        print("Step: {}\t||\tAccuracy: {:.4}\t||\tLoss: {:.4}".format(i, acc, loss))

        if(i%100==0):
            # print("\nEval:")
            # weights = [v for v in tf.trainable_variables() if "conv-layer-3/W" in v.name]
            # print(session.run(weights,feed_dict=feed_dict))
            feed_dict_test = {x:test_x, y:test_y,dropoutprob:1.0}
            testacc=session.run(accuracy, feed_dict=feed_dict_test)
            print("\nEval:||\tAccuracy: {}".format(testacc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time
    print("Time Used:",time_dif)


session = tf.Session()
# session = tf_debug.LocalCLIDebugWrapperSession(session)
# session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
session.run(tf.global_variables_initializer())

# feed_dict = {x:xbatch1, y:ybatch1}
# cls_pred = session.run(predictions, feed_dict=feed_dict)
# print(cls_pred)

batches = dataprep.batch_iter(list(zip(train_x,train_y)), 100,200)
# print(len(list(batches)))
train_network(batches)

# session.close()
