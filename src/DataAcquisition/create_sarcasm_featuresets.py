
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import json
from collections import Counter

lemmatizer = WordNetLemmatizer()

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
    return l2


def createFeaturesFromData(sample, lexicon):
    featureset=[]
    with open(sample, 'r') as f:
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
                featureset.append([features,[1,0]])
            elif(label=='0'):
                featureset.append([features,[0,1]])
            else:
                print('No Label')
    return featureset
# lex=create_lexicon('/Users/FelixDSantos/LeCode/DeepLearning/fyp/SarcasmDataset_Final.txt')
#
# fset= sample_handling('/Users/FelixDSantos/LeCode/DeepLearning/fyp/SarcasmDataset_Final.txt',lex)

def CreateTweetTrainAndTest(sarcasmset,test_size = 0.1):
    lexicon = create_lexicon(sarcasmset)
    features=[]
    features+=createFeaturesFromData(sarcasmset,lexicon)
    random.shuffle(features)
    features=np.array(features)

    # features are in the shape [[[0,1,1,1,0],[1,0]],[[1,1,1,0,0],[0,1]]]
    #  [features,label]
    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])

    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x,train_y,test_x,test_y
#


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


# ashwin dataset:
sarcasmdataset='/Users/FelixDSantos/LeCode/DeepLearning/fyp/Data/Cleaned/SarcasmDataset_Final.txt'
if __name__ == '__main__':
    train_x,train_y,test_x,test_y = CreateTweetTrainAndTest(sarcasmdataset)
    with open('/Users/FelixDSantos/LeCode/DeepLearning/fyp/TrainAndTest/sentiment_set_old.json', 'w') as outfile:
        json.dump([train_x,train_y,test_x,test_y], outfile)
