import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.interpolate import spline
from scipy.interpolate import interp1d
import argparse
import sys
sys.path.append('/Users/FelixDSantos/LeCode/DeepLearning/fyp/src/DataAcquisition')
import create_sarcasm_featuresets as dataprep
# import DataAcquisition.create_sarcasm_featuresets as dataprep
import numpy as np
from termcolor import colored
# trainresults='/Users/FelixDSantos/LeCode/DeepLearning/fyp/Results/DeepNeuralNetwork/train_2layers.csv'
# testresults = '/Users/FelixDSantos/LeCode/DeepLearning/fyp/Results/DeepNeuralNetwork/test_2_layers.csv'
# TODO: use savitsky golay filter smoothing
resultsdir = os.path.abspath("../../Results/")
classdic={'pos':'1','neg':'0'}
"""
Result Files for evaluation of neural network for Ashwin Dataset
"""
def getparams():
    parser = argparse.ArgumentParser(description='Learning curve Args')
    parser.add_argument('-trainres', action='store', dest='trainresults',
                        help='Store a Train Results path')
    parser.add_argument('-testres', action='store', dest='testresults',
                        help='Store a Test Results path')
    parser.add_argument('-plottitle', action='store', dest='plottitle',
                        help='Store a Learning Curve Plot name')
    parser.add_argument('-savedname', action='store', dest='savedname',
                        help='Store saved name of Learning curve')

    plotargs = parser.parse_args()
    trainresults=plotargs.trainresults
    testresults = plotargs.testresults
    plottitle= plotargs.plottitle
    savedname=plotargs.savedname
    return(trainresults,testresults,plottitle,savedname)

def printc(text,color=None,attributes=[]):
    print(colored(text,color,attrs=attributes))
def plot_conf_matrix(y_true, y_pred):

    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    empty=pd.DataFrame(np.zeros((3,3)))
    # empty.columns = ['0','1','All']
    confmatrix=pd.crosstab(y_true,y_pred, rownames = ['True'], colnames=['Predicted'],margins=True)
    confmatrixadded = empty.add(confmatrix,fill_value=0).fillna(0)
    confmatrixadded=confmatrixadded.drop(confmatrixadded.index[2]).drop(2,axis=1).astype(int)
    # confmatrix.columns=['0','1','All']
    # confmatrixem=confmatrix.add(empty)
    # confmatrix=confmatrix.add(empty)

    # confmatrix.index=['0','1','All']
    # print("============================================Confusion_matrix=======================================")
    # print(confmatrixadded)
    printc("=====================================Confusion Matrix=================================================================",attributes=['bold'])
    print(confmatrixadded)
    printc("======================================================================================================================",attributes=['bold'])
    # print("====================================================================================================")

    return(confmatrixadded)

def showclassifiedexamples(data,ytrue,ypred,num=10,correct=False):
    boolpred=(ytrue==ypred)
    wrongclassified=np.where((boolpred==[correct]))[0]
    indexofwrong=np.unique(wrongclassified)
    datawrongpred=np.take(data,indexofwrong)
    trueclassofwrong,predictedclassofwrong=ytrue[indexofwrong],ypred[indexofwrong]

    labels={1:"Non-Sarcastic",0:"Sarcastic"}

    for indx in range(0,num):
        true=labels.get(trueclassofwrong[indx])
        pred=labels.get(predictedclassofwrong[indx])

        print("Tweet: ",datawrongpred[indx])
        print("True Class: ",true)
        print("Predicted Class: ",pred)
        if(correct):
            printc("======================================================================================================================\n","green")
        else:
            printc("======================================================================================================================\n","red")
# data=dataprep.loaddatafromjson("/Users/FelixDSantos/LeCode/DeepLearning/fyp/FeatureData/Holdout/tweetslabelsAndHoldOut_Ash")
# x=data[2]
# ytrue=np.array(data[3])
# np.random.seed(2)
# shuffleindx = np.random.permutation(np.arange(len(ytrue)))
# ypred=ytrue[shuffleindx]
#
# showwrongclassified(x,ytrue,ypred)
# print(data[0][0])
# y_true = np.array([0,0,0,1,1,0])
# y_pred = np.array([0,0,0,0,0,0])
# plot_conf_matrix(y_true,y_pred)
def plotlearningcurve(trainresults,testresults,getevery,TitleOfPlot,savedname,savePlotToFile=False):
    traindatares=pd.read_csv(trainresults,header=0,names=['Time','Step','Value'])
    testdatares=pd.read_csv(testresults,header=0,names=['Time','Step','Value'])
    traindatares= traindatares.iloc[::getevery,:]
    testres=testdatares.as_matrix(columns=testdatares.columns[1:])
    trainres=traindatares.as_matrix(columns=traindatares.columns[1:])
    xtrain,ytrain = trainres[:,0],trainres[:,1]
    xtest,ytest=testres[:,0],testres[:,1]
    x_smoothed = np.linspace(xtrain.min(),xtrain.max(),20)
    y_train_smooth=spline(xtrain,ytrain,x_smoothed)
    f = interp1d(xtrain, ytrain,kind ='cubic')
    ftest = interp1d(xtest,ytest)
    plt.title(TitleOfPlot, y=1.08,fontsize=15)
    plt.plot(x_smoothed,f(x_smoothed),'r',label="Train Accuracy")
    plt.plot(x_smoothed,ftest(x_smoothed), 'b',label="Test Accuracy")
    plt.xlabel('Number of Batches')
    plt.ylabel('Accuracy')
    leg=plt.subplot()
    leg.plot()
    box = leg.get_position()
    leg.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    leg.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    outpath=os.path.join(resultsdir,"DeepNeuralNetwork","Plots","Learning_Curve")
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    if(savePlotToFile==True):
        plotsavedir= os.path.join(outpath,savedname)
        plt.savefig(plotsavedir,bbox_inches='tight')
    plt.show()

def calculaterecall(confmatrix,classval='pos'):
    val=classdic.get(classval)
    numtrueclass=confmatrix.loc[val][val]
    numAllTrue=confmatrix.loc[val]['All']
    recall = (numtrueclass)/(numAllTrue)
    return(recall)

def calculateprecision(confmatrix,classval='pos'):
    val=classdic.get(classval)
    numtrueclass=confmatrix.loc[val][val]
    numClassedAsclass=confmatrix.loc['All'][val]
    precision= numtrueclass/numClassedAsclass
    return(precision)

def calculateF1(r,p):
    f1= 2*((p*r)/(p+r))
    return(f1)

def calculateModelStats(confmatrix):
    r=calculaterecall(confmatrix)
    p=calculateprecision(confmatrix)
    F1=calculateF1(r,p)
    printc("===================================Model Stats For Classifying Sarcasm================================================",attributes=["bold"])
    print('Recall:{}'.format(r))
    print('Precision:{}'.format(p))
    print('F1 Score:{}'.format(F1))
    printc("======================================================================================================================",attributes=["bold"])
    rneg=calculaterecall(confmatrix,classval='neg')
    pneg=calculateprecision(confmatrix,classval='neg')
    F1neg=calculateF1(rneg,pneg)
    printc("===================================Model Stats For Classifying Not-Sarcasm============================================",attributes=["bold"])
    print('Recall:{}'.format(rneg))
    print('Precision:{}'.format(pneg))
    print('F1 Score:{}'.format(F1neg))
    printc("=======================================================================================================================",attributes=["bold"])
# if __name__ == '__main__':
#     # Plot learning curve for neural network with 1 layers 500 neurons on ashwin Dataset
#     trainresults,testresults,plottitle,savedname = getparams()
#     plotlearningcurve(trainresults,testresults,getevery = 50,
#                     savePlotToFile=True,TitleOfPlot=plottitle,
#                     savedname=savedname)
