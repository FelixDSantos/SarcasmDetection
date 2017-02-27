import numpy as np
import pandas as pd
# TODO put code here for confusion matrix

def plot_conf_matrix(y_true, y_pred):

    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    confmatrix=pd.crosstab(y_true,y_pred, rownames = ['True'], colnames=['Predicted'],margins=True)

    print("============================================Confusion_matrix=======================================")
    print(confmatrix)
    print("====================================================================================================")
