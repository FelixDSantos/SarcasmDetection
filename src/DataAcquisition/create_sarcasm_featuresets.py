
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import json
from collections import Counter
import argparse
from tensorflow.contrib import learn

lemmatizer = WordNetLemmatizer()
# Paramaters
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
    # we don't want to use words like "the"
    for w in w_counts:
        if 1500>w_counts[w]>50:
            l2.append(w)
    print("Lexicon created with vocabulary size {}".format(len(l2)))
    return l2


def createFeaturesFromData(tweetx,tweety, lexicon,header=True):
    featureset=[]
    for index, item in enumerate(tweetx):
        current_words=word_tokenize(tweetx[index].lower())
        label=tweety[index]
        current_words=[lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))
        for word in current_words:
            if word.lower() in lexicon:
                index_value=lexicon.index(word.lower())
                features[index_value]+= 1
        features=list(features)
        featureset.append([features,label])
    return featureset

def prepdata(file):
    '''
    This method is used alternatievly to the above method. It does not create feature vectors.
    It only creates two lists. A list for the tweets and a list for the labels
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

def CreateTweetFeatures(datax,datay,lexicon):
    features=[]
    features+=createFeaturesFromData(datax,datay,lexicon)
    features=np.array(features)
    x=features[:,0]
    y=features[:,1]
    print("{} Tweet Features Created".format(len(y)))
    return list(x),list(y)
#
def partitionDataToTrainandTest(x,y,lenwholeset,trainingpercent):
    trainingsize=np.floor((trainingpercent/100)*(lenwholeset)).astype(int)
    shufflindx=np.random.permutation(np.arange(len(y)))
    # y=np.asarray(y)
    x,y=x[shufflindx],y[shufflindx]
    x_train , y_train = x[0:trainingsize], y[0:trainingsize]
    x_test , y_test = x[trainingsize:len(y)], y[trainingsize:len(y)]
    return(x_train,y_train,x_test,y_test)

def holdoutdata(x,y,holdoutpercent,shuffle=True,defaultcheck=True):
    validationsize = np.floor((holdoutpercent/100)*len(y)).astype(int)
    # x,y=np.asarray(x),np.asarray(y)
    x,y=np.array(x),np.array(y)
    if(shuffle):
        shuffleindx = np.random.permutation(np.arange(len(y)))
        x,y=x[shuffleindx],y[shuffleindx]
    x,y = (x[0:(len(y)-validationsize)]).tolist(),(y[0:(len(y)-validationsize)]).tolist()
    x_val,y_val =(x[(len(y)-validationsize):len(y)]),(y[(len(y)-validationsize):len(y)])
    return(x,y,x_val,y_val)

def batch_iter(data, batch_size , num_epochs , shuffle = True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size)+1
    print("Num of batches per epoch: ",num_batches_per_epoch)
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

def buildvocab(tweets,holdouttweets):
    # this takes each word in dataset and changes it to an integer
    # creates a vocabulary for each word
    # no lemmatizing or any other word preprocessing done
    # returns list of tweet features and a vocabulary
    wholeset=tweets+holdouttweets
    maxtweetlength = max([len(tweet.split(" ")) for tweet in wholeset])
    vocabproc = learn.preprocessing.VocabularyProcessor(maxtweetlength)

    # creating a vocabulary of the tweets as well as making features of length-max_document_length out of each tweet
    tweet_feats = (np.array(list(vocabproc.fit_transform(tweets)))).tolist()
    hold_feats = (np.array(list(vocabproc.fit_transform(holdouttweets)))).tolist()
    vocabsize = len(vocabproc.vocabulary_)
    print("Vocabulary Size: ",vocabsize)
    print("{} tweet features created.".format(len(tweet_feats)))
    print("{} Hold out tweet features created.".format(len(hold_feats)))
    return(vocabsize,tweet_feats,hold_feats)


def getparamsfornn():
    parser = argparse.ArgumentParser(description='Create Features')
    parser.add_argument('-network', action='store', dest='networktype',default='c',help='Features to be tailored for networktype')
    parser.add_argument('-s', action='store', dest='sarcasmdataset',default='a',
                        help='Choose Sarcasm Dataset for Training- a:ashwin,b:bamman and smith,ab: Both Ash and bamman data')
    parser.add_argument('-l', action='store', dest='lexicondataset',default='a',
                        help='Choose Dataset for Creating lexicon- a:ashwin,b:bamman and smith,ab: Both Ash and bamman data')
    parser.add_argument('-o', action='store', dest='output',default='../../FeatureData/features.json',help='Outputlocation for features json.')
    parser.add_argument('-hold', action='store', dest='holdoutpercent',default='10',help='The percentage of data to hold out.')
    args = parser.parse_args()
    networktype=args.networktype
    datachoice=args.sarcasmdataset
    outputlocation=args.output
    holdoutpercent=float(args.holdoutpercent)
    lexchoice=args.lexicondataset
    datasets={'a':ashwinsarcasmdataset,'b':bMssarcasmdataset,'ab':ABmssarcasmdataset}

    datasetloc=datasets.get(datachoice)
    lexicondataset = datasets.get(lexchoice)

    print("Lexicon Dataset from:{}".format(lexicondataset))
    print("Sarcasm Dataset For training:{}".format(datasetloc))
    return(networktype,datasetloc,outputlocation,lexicondataset,holdoutpercent)

def loaddatafromjson(path):
    with open(path) as openfile:
            data = json.load(openfile)
            tweets,labels,heldout_x,heldout_y = data[0],data[1],data[2],data[3]

            return tweets,labels,heldout_x,heldout_y

#
if __name__ == '__main__':
    # train_x,train_y,test_x,test_y = CreateTweetTrainAndTest(sarcasmdataset)
    networktype,sarcasmdataset,outputlocation,lexiconloc,holdoutperc=getparamsfornn()
    # ==============================================================================
    # UNCOMMENT IF YOU WANT TO HOLD OUT DATASET AND SAVE TO JSON
    # holdoutdata(dataloc=sarcasmdataset,holdoutpercent=holdoutperc)
    # ==============================================================================
    if(networktype=='d'):
        lexicon=create_lexicon(lexiconloc)
        tweetdata,tweetlabels,held_out_x,held_out_y=loaddatafromjson("/Users/FelixDSantos/LeCode/DeepLearning/fyp/FeatureData/Holdout/tweetslabelsAndHoldOut_bms")
        held_outxfeats,held_outyfeats=CreateTweetFeatures(held_out_x,held_out_y,lexicon)
        tweetx,tweety=CreateTweetFeatures(tweetdata,tweetlabels,lexicon)
        print("Held Out dataset {} Features".format(len(held_outyfeats)))
        with open(outputlocation, 'w') as outfile:
            json.dump([lexicon,tweetx,tweety,held_outxfeats,held_outyfeats], outfile)
            print("File written to ", outputlocation)
    elif(networktype=='c'):
        tweetdata,tweetlabels,held_out_x,held_out_y=loaddatafromjson("/Users/FelixDSantos/LeCode/DeepLearning/fyp/FeatureData/Holdout/tweetslabelsAndHoldOut_Ash")
        vocabsize,tweet_x,held_out_xfeats=buildvocab(tweetdata,held_out_x)
        print("Rest of Data: {} Features".format(len(tweetlabels)))
        print("Held Out dataset {} Features".format(len(held_out_y)))
        with open(outputlocation, 'w') as outfile:
            json.dump([vocabsize,tweet_x,tweetlabels,held_out_xfeats,held_out_y], outfile)
            print("File written to ", outputlocation)
