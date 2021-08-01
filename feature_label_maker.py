import numpy as np
import pandas as pd
import constants as constx

def label_binary(dfx):
    labels1       = dfx.loc[:,constx.ret_future].shift(0)/dfx.loc[:,constx.ret_curr].shift(0) -1
    labels2       = np.where(labels1>0,1,-1)
    frame         = pd.DataFrame({'label' : labels2,'future_ret':labels1})
    frame         = frame.set_index(dfx.index)
    frame.columns = ['label','future_ret']
    return (frame)

def label_binary2(dfx):
    labels1       = dfx.loc[:,constx.ret_future].shift(0)/dfx.loc[:,constx.ret_curr].shift(0) -1
    labels2       = np.where(labels1>0,1,-1)
    labels2       = np.where(labels2 == -1,0,1)
    
    frame         = pd.DataFrame({'label' : labels2,'future_ret':labels1})
    frame         = frame.set_index(dfx.index)
    frame.columns = ['label','future_ret']
    return (frame)


def label_multiple(dfx ,thresh):
    labels1       = dfx.loc[:,constx.ret_future].shift(0)/dfx.loc[:,constx.ret_curr].shift(0) -1
    labels2       = np.where(labels1>thresh,1,np.where(labels1< -thresh,-1,0))
    frame         = pd.DataFrame({'label' : labels2,'future_ret':labels1})
    frame         = frame.set_index(dfx.index)
    frame.columns = ['label','future_ret']
    return (frame)

def label_regress(dfx ,thresh):
    labels1       = dfx.loc[:,constx.ret_future].shift(0)/dfx.loc[:,constx.ret_curr].shift(0) -1
    labels2       = np.where(labels1>thresh,1,np.where(labels1< -thresh,-1,0))
    frame         = pd.DataFrame({'label' : labels1,'future_ret':labels1})
    frame         = frame.set_index(dfx.index)
    frame.columns = ['label','future_ret']
    return (frame)

def load_all_feature_types(dfx):
    
    feat1 = dfx.loc[:,'SUBJECT']
    feat2 = dfx.loc[:,'DETAILS']
    frame         = pd.DataFrame({'subject' : feat1,'details':feat2})
    frame         = frame.set_index(dfx.index)
    frame.columns = ['subject','details']
    
    return (frame)

def load_all_feature_types2(dfx):
    
    feat2 = dfx.loc[:,'TEXT']
    frame         = pd.DataFrame({'details':feat2})
    frame         = frame.set_index(dfx.index)
    frame.columns = ['details']
    
    return (frame)


    
    #smooth_features     = smoothing_features(dfx)

