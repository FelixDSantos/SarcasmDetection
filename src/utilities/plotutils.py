import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.interpolate import spline
from scipy.interpolate import interp1d
import argparse
# trainresults='/Users/FelixDSantos/LeCode/DeepLearning/fyp/Results/DeepNeuralNetwork/train_2layers.csv'
# testresults = '/Users/FelixDSantos/LeCode/DeepLearning/fyp/Results/DeepNeuralNetwork/test_2_layers.csv'
# TODO: use savitsky golay filter smoothing
resultsdir = os.path.abspath("../../Results/")

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
def plot_conf_matrix(y_true, y_pred):

    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    confmatrix=pd.crosstab(y_true,y_pred, rownames = ['True'], colnames=['Predicted'],margins=True)

    print("============================================Confusion_matrix=======================================")
    print(confmatrix)
    print("====================================================================================================")

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

if __name__ == '__main__':
    # Plot learning curve for neural network with 1 layers 500 neurons on ashwin Dataset
    trainresults,testresults,plottitle,savedname = getparams()
    plotlearningcurve(trainresults,testresults,getevery = 50,
                    savePlotToFile=True,TitleOfPlot=plottitle,
                    savedname=savedname)
