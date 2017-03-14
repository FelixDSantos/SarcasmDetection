
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import json
from collections import Counter
import argparse
lemmatizer = WordNetLemmatizer()
# ashwin dataset:
ashwinsarcasmdataset='/Users/FelixDSantos/LeCode/DeepLearning/fyp/Data/Cleaned/SarcasmDataset_Final.txt'
# Bamman and Smith
bMssarcasmdataset = '/Users/FelixDSantos/LeCode/DeepLearning/fyp/BnSData/SarcasmDataset_Final.txt'
# both datasets together
ABmssarcasmdataset='/Users/FelixDSantos/LeCode/DeepLearning/fyp/BnsAndAsh/bnsandash.txt'
def create_lexicon(sarcasmset):
    # create a lexicon with all strings in positive and negative datasets
    lexicon=[]
    for fi in[sarcasmset]:
        with open(fi, 'r') as f:
            header = next(f)
            # contents = f.readlines()
            for l in f:
                all_words = word_tokenize(l[:-2].lower())
                lexicon+=list(all_words)
    # lematize all words in the lexicon
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    # create a dictionary object of counts
    w_counts= Counter(lexicon)
    l2=[]
    # l2 is the final lexicon
    # we don't want to use words like "the", "and and such"
    for w in w_counts:
        if 1500>w_counts[w]>50:
            l2.append(w)
    print("Lexicon created with vocabulary size {}".format(len(l2)))
    return l2


def createFeaturesFromData(data, lexicon,header=True):
    featureset=[]
    with open(data, 'r') as f:
        if(header==True):
            header = next(f)
        # contents= f.readlines()
        for l in f:
            fullline=word_tokenize(l.lower())
            label=fullline[len(fullline)-1]
            current_words=fullline[:-1]
            current_words=[lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value=lexicon.index(word.lower())
                    features[index_value]+= 1
            features=list(features)
            if(label=='1'):
                featureset.append([features,[0,1]])
            elif(label=='0'):
                featureset.append([features,[1,0]])
            else:
                print('No Label')
    return featureset

def prepdata(file):
    '''
    This method is used alternatievly to the above method. It does not create feature vectors.
    It only creates two lists. A list for the tweets and a list for the data
    '''
    tweet_x=[]
    tweet_y=[]
    with open(file , 'r') as f:
        header = next(f)
        for l in f:
            l= l.strip().lower()
            tweet=l[0:len(l)-3]
            label = l[len(l)-1]

            if(label == '1'):
                tweet_x.append(tweet)
                tweet_y.append([1,0])
            elif(label == '0'):
                tweet_x.append(tweet)
                tweet_y.append([0,1])
            else:
                print("No Label")
    return tweet_x,tweet_y


# prepdata('/Users/FelixDSantos/LeCode/DeepLearning/fyp/Data/Cleaned/SarcasmDataset_Final.txt')
# lex=create_lexicon('/Users/FelixDSantos/LeCode/DeepLearning/fyp/SarcasmDataset_Final.txt')
#
# fset= sample_handling('/Users/FelixDSantos/LeCode/DeepLearning/fyp/SarcasmDataset_Final.txt',lex)

def CreateTweetFeatures(sarcasmset,lexicon):
    # lexicon = create_lexicon(sarcasmset)
    features=[]
    features+=createFeaturesFromData(sarcasmset,lexicon)
    # random.shuffle(features)
    features=np.array(features)

    # features are in the shape [[[0,1,1,1,0],[1,0]],[[1,1,1,0,0],[0,1]]]
    #  [features,label]
    # testing_size = int(test_size*len(features))

    x=features[:,0]
    y=features[:,1]
    # train_x = list(features[:,0][:-testing_size])
    # train_y = list(features[:,1][:-testing_size])
    #
    # test_x = list(features[:,0][-testing_size:])
    # test_y = list(features[:,1][-testing_size:])
    print("{} Tweet Features Created".format(len(y)))
    return x,y
#
def partitionDataToTrainandTest(x,y,lenwholeset,trainingpercent):
    trainingsize=np.floor((trainingpercent/100)*(lenwholeset)).astype(int)
    shufflindx=np.random.permutation(np.arange(len(y)))
    # y=np.asarray(y)
    x,y=x[shufflindx],y[shufflindx]
    x_train , y_train = x[0:trainingsize], y[0:trainingsize]
    x_test , y_test = x[trainingsize:len(y)], y[trainingsize:len(y)]
    return(x_train,y_train,x_test,y_test)

def holdoutdata(x,y,holdoutpercent,shuffle=True):
    validationsize = np.floor((holdoutpercent/100)*len(y)).astype(int)
    # x,y=np.asarray(x),np.asarray(y)
    x,y=np.array(x),np.array(y)
    if(shuffle):
        shuffleindx = np.random.permutation(np.arange(len(y)))
        x,y=x[shuffleindx],y[shuffleindx]
    x,y = list(x[0:(len(y)-validationsize)]),list(y[0:(len(y)-validationsize)])
    x_val,y_val =list(x[(len(y)-validationsize):len(y)]),list(y[(len(y)-validationsize):len(y)])

    return(x,y,x_val,y_val)
def batch_iter(data, batch_size , num_epochs , shuffle = True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size)+1

    for epoch in range(num_epochs):
        #shuffle the data at each epoch

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]

        else:
            shuffled_data=data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1)*batch_size, data_size)
            yield shuffled_data[start_index:end_index]
def getparams():
    parser = argparse.ArgumentParser(description='Create Features')
    parser.add_argument('-s', action='store', dest='sarcasmdataset',
                        help='Choose Sarcasm Dataset for Training- a:ashwin,b:bamman and smith,ab: Both Ash and bamman data')
    parser.add_argument('-l', action='store', dest='lexicondataset',
                        help='Choose Dataset for Creating lexicon- a:ashwin,b:bamman and smith,ab: Both Ash and bamman data')
    parser.add_argument('-o', action='store', dest='output',default='../../FeatureData/features.json',help='Outputlocation for features json.')
    parser.add_argument('-hold', action='store', dest='holdoutpercent',default='10',help='The percentage of data to hold out.')
    args = parser.parse_args()
    datachoice=args.sarcasmdataset
    outputlocation=args.output
    holdoutpercent=float(args.holdoutpercent)
    lexchoice=args.lexicondataset
    datasets={'a':ashwinsarcasmdataset,'b':bMssarcasmdataset,'ab':ABmssarcasmdataset}

    datasetloc=datasets.get(datachoice)
    lexicondataset = datasets.get(lexchoice)

    print("Lexicon Dataset from:{}".format(lexicondataset))
    print("Sarcasm Dataset For training:{}".format(datasetloc))
    return(datasetloc,outputlocation,lexicondataset,holdoutpercent)

if __name__ == '__main__':
    # train_x,train_y,test_x,test_y = CreateTweetTrainAndTest(sarcasmdataset)
    sarcasmdataset,outputlocation,lexiconloc,holdoutperc=getparams()
    lexicon=create_lexicon(lexiconloc)
    features,labels= CreateTweetFeatures(sarcasmdataset,lexicon)
    features,labels, held_outfeatures,heldout_labels=holdoutdata(features,labels,holdoutpercent=holdoutperc)
    print("Held Out dataset {} Features".format(len(heldout_labels)))
    with open(outputlocation, 'w') as outfile:
        json.dump([lexicon,features,labels,held_outfeatures,heldout_labels], outfile)
        print("File written to ", outputlocation)
