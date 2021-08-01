
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree     import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score

import constants as constx

import pandas as pd
import numpy as np
from scipy.ndimage.interpolation import shift
import random

from xgboost import plot_tree
import matplotlib.pyplot as plt



def Adaboostclass(train_features,train_labels,validate_features,validate_labels,train_future_ret,valid_future_ret,features):
    
    base_model = DecisionTreeClassifier(max_depth=1 ,class_weight="balanced",
                                        min_samples_split = 0.15,min_samples_leaf = 0.05)
        
    clf1       = AdaBoostClassifier(base_model,n_estimators= 50 ,learning_rate= 0.25)
    clf        = clf1.fit(train_features, train_labels)
    importance = clf.feature_importances_
    print("\n", " AdaBoost Classifier \n") 

    predictions = clf.predict(train_features)
    conf_mat = confusion_matrix(train_labels, predictions)
    print(conf_mat)
    print(classification_report(train_labels, predictions))
    
    predictions2 = clf.predict(validate_features)
    conf_mat = confusion_matrix(validate_labels, predictions2)
    print(conf_mat)
    print(classification_report(validate_labels,predictions2))
    
    len_train     = len(train_features)
    y_pred_train  = clf.predict(train_features)
    tc_pred_train = np.where(y_pred_train != shift(y_pred_train, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_train     = np.where(y_pred_train == 1, train_future_ret - tc_pred_train,np.where(y_pred_train == -1,-train_future_ret - tc_pred_train,0))
    
    
    print("total Return ",np.mean(ret_train)*100*252,"%")
    print("total tc ",sum(tc_pred_train)*100,"%")
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(252)
    print("sharpe_train",sharpe_train )
    
    len_valid    = len(validate_features)
    y_pred_valid = clf.predict(validate_features)
    tc_pred_valid = np.where(y_pred_valid != shift(y_pred_valid, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_valid    = np.where(y_pred_valid == 1, valid_future_ret - tc_pred_valid,np.where(y_pred_valid == -1, -valid_future_ret - tc_pred_valid,0))
    
    print("total Return ",np.mean(ret_valid)*100*252,"%")
    print("total tc ",sum(tc_pred_valid)*100,"%")
    sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(252)
    print("sharpe_valid",sharpe_valid)
    
    
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(25,25)) 
    plt.title('Adaboost Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    return clf,y_pred_train,y_pred_valid


def RandomForestClass(train_features,train_labels,validate_features,validate_labels,train_future_ret,valid_future_ret,features):
    
    wts2  = np.abs(train_future_ret)*100
    
    rfc1 = RandomForestClassifier(n_estimators = 20,max_depth = 2,max_features= 0.67,
                                  class_weight= 'balanced_subsample',random_state=11 ,
                                  min_samples_split = 0.05,min_samples_leaf=0.05)
                                  #min_weight_fraction_leaf= 0.1 ,min_impurity_decrease = 0.05)
    
    rfc  = rfc1.fit(train_features,train_labels)    
    
    importance = rfc.feature_importances_
    print("\n", " RandomForest Classifier \n ") 
    
    predictions  = rfc.predict(train_features)   
    conf_mat = confusion_matrix(train_labels, predictions)
    print(conf_mat)
    print(classification_report(train_labels, predictions))
    
    predictions2 = rfc.predict(validate_features)   
    conf_mat = confusion_matrix(validate_labels, predictions2)
    print(conf_mat)
    print(classification_report(validate_labels,predictions2))
    
    len_train     = len(train_features)
    y_pred_train  = rfc.predict(train_features)
    tc_pred_train = np.where(y_pred_train != shift(y_pred_train, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_train     = np.where(y_pred_train == 1, train_future_ret - tc_pred_train,np.where(y_pred_train == -1,-train_future_ret - tc_pred_train,0))
    
    
    print("total Return ",np.mean(ret_train)*100*252,"%")
    print("total tc ",sum(tc_pred_train)*100,"%")
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(252)
    print("sharpe_train",sharpe_train )
    
    len_valid    = len(validate_features)
    y_pred_valid = rfc.predict(validate_features)
    tc_pred_valid = np.where(y_pred_valid != shift(y_pred_valid, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_valid    = np.where(y_pred_valid == 1, valid_future_ret - tc_pred_valid,np.where(y_pred_valid == -1, -valid_future_ret - tc_pred_valid,0))
    
    print("total Return ",np.mean(ret_valid)*100*252,"%")
    print("total tc ",sum(tc_pred_valid)*100,"%")
    sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(252)
    print("sharpe_valid",sharpe_valid)
    
    
    
    importances = rfc.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(25,25)) 
    plt.title('RandomForest Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    return rfc,y_pred_train,y_pred_valid


def NeuralNetworkClass(train_features,train_labels,validate_features,validate_labels,train_future_ret,valid_future_ret,features):
    
    mlp1 = MLPClassifier(hidden_layer_sizes=(10,10,10,10),max_iter=2000,activation = 'relu',
                         alpha = 0.01,solver = 'adam',learning_rate= 'adaptive')
    
    mlp  = mlp1.fit(train_features, train_labels)
    
    print("\n", " NeuralNetwork Classifier")
    
    predictions = mlp.predict(train_features)
    conf_mat = confusion_matrix(train_labels, predictions)
    print(conf_mat)
    print(classification_report(train_labels, predictions))
    
    predictions2 = mlp.predict(validate_features)
    conf_mat = confusion_matrix(validate_labels, predictions2)
    print(conf_mat)
    print(classification_report(validate_labels,predictions2))
    
    len_train     = len(train_features)
    y_pred_train  = mlp.predict(train_features)
    tc_pred_train = np.where(y_pred_train != shift(y_pred_train, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_train     = np.where(y_pred_train == 1, train_future_ret - tc_pred_train,np.where(y_pred_train == -1,-train_future_ret - tc_pred_train,0))
    
    
    print("total Return ",np.mean(ret_train)*100*252,"%")
    print("total tc ",sum(tc_pred_train)*100,"%")
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(252)
    print("sharpe_train",sharpe_train )
    
    len_valid     = len(validate_features)
    y_pred_valid  = mlp.predict(validate_features)
    tc_pred_valid = np.where(y_pred_valid != shift(y_pred_valid, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_valid    = np.where(y_pred_valid == 1, valid_future_ret - tc_pred_valid,np.where(y_pred_valid == -1, -valid_future_ret - tc_pred_valid,0))
    
    print("total Return ",np.mean(ret_valid)*100*252,"%")
    print("total tc ",sum(tc_pred_valid)*100,"%")
    sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(252)
    print("sharpe_valid",sharpe_valid)
    
        
    return mlp


def XGBoostClass(train_features,train_labels,validate_features,validate_labels,train_future_ret,valid_future_ret,features):

    wts2  = np.abs(train_future_ret)
    wts2  = wts2*wts2*100
    xgbc1 = XGBClassifier(n_estimators = 50,max_depth = 3,eta = 0.1,
                          gamma = 0.0,subsample = 0.6)
    
    xgbc  = xgbc1.fit(train_features, train_labels)
    
    importance = xgbc.feature_importances_
    print("\n", " XGBoost Classifier") 
    
    predictions = xgbc.predict(train_features)
    conf_mat = confusion_matrix(train_labels, predictions)
    print(conf_mat)
    print(classification_report(train_labels, predictions))
    
    predictions2 = xgbc.predict(validate_features)
    conf_mat = confusion_matrix(validate_labels, predictions2)
    print(conf_mat)
    print(classification_report(validate_labels,predictions2))
    
    len_train     = len(train_features)
    y_pred_train  = xgbc.predict(train_features)
    tc_pred_train = np.where(y_pred_train != shift(y_pred_train, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_train     = np.where(y_pred_train == 1, train_future_ret - tc_pred_train,np.where(y_pred_train == -1,-train_future_ret - tc_pred_train,0))
    
    
    print("total Return ",np.mean(ret_train)*100*252,"%")
    print("total tc ",sum(tc_pred_train)*100,"%")
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(252)
    print("sharpe_train",sharpe_train )
    
    len_valid    = len(validate_features)
    y_pred_valid = xgbc.predict(validate_features)
    tc_pred_valid = np.where(y_pred_valid != shift(y_pred_valid, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_valid    = np.where(y_pred_valid == 1, valid_future_ret - tc_pred_valid,np.where(y_pred_valid == -1, -valid_future_ret - tc_pred_valid,0))
    
    print("total Return ",np.mean(ret_valid)*100*252,"%")
    print("total tc ",sum(tc_pred_valid)*100,"%")
    sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(252)
    print("sharpe_valid",sharpe_valid)
    
    
    
    importances = xgbc.feature_importances_
    indices = np.argsort(importances)[(len(importances) - 10):len(importances)]
    
    # plt.figure(figsize=(10,10)) 
    # plt.title('XGboost Feature Importances')
    # plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    # plt.yticks(range(len(indices)), [features[i] for i in indices])
    # plt.xlabel('Relative Importance')
    # plt.show()
        
    return xgbc,y_pred_train,y_pred_valid


def Bagclass(train_features,train_labels,validate_features,validate_labels,train_future_ret,valid_future_ret,features):
    
    base_model = MLPClassifier(hidden_layer_sizes=(5,5),max_iter=500,activation = 'relu',
                         alpha = 0.0001,solver = 'sgd',learning_rate= 'adaptive' ,
                         tol = 0.01)
        
    clf1       = BaggingClassifier(base_model, n_estimators= 20)
    clf        = clf1.fit(train_features, train_labels)
    #importance = clf.feature_importances_
    print("\n", " AdaBoost Classifier \n") 

    predictions = clf.predict(train_features)
    conf_mat = confusion_matrix(train_labels, predictions)
    print(conf_mat)
    print(classification_report(train_labels, predictions))
    
    predictions2 = clf.predict(validate_features)
    conf_mat = confusion_matrix(validate_labels, predictions2)
    print(conf_mat)
    print(classification_report(validate_labels,predictions2))
    
    len_train     = len(train_features)
    y_pred_train  = clf.predict(train_features)
    tc_pred_train = np.where(y_pred_train != shift(y_pred_train, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_train     = np.where(y_pred_train == 1, train_future_ret - tc_pred_train,np.where(y_pred_train == -1,-train_future_ret - tc_pred_train,0))
    
    
    print("total Return ",np.mean(ret_train)*100*252,"%")
    print("total tc ",sum(tc_pred_train)*100,"%")
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(252)
    print("sharpe_train",sharpe_train )
    
    len_valid    = len(validate_features)
    y_pred_valid = clf.predict(validate_features)
    tc_pred_valid = np.where(y_pred_valid != shift(y_pred_valid, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_valid    = np.where(y_pred_valid == 1, valid_future_ret - tc_pred_valid,np.where(y_pred_valid == -1, -valid_future_ret - tc_pred_valid,0))
    
    print("total Return ",np.mean(ret_valid)*100*252,"%")
    print("total tc ",sum(tc_pred_valid)*100,"%")
    sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(252)
    print("sharpe_valid",sharpe_valid)
    
        
    return clf,y_pred_train,y_pred_valid


def Stackclass(train_features,train_labels,validate_features,validate_labels,train_future_ret,valid_future_ret,features):
    
    estimators = [
        ('rf1w',  RandomForestClassifier(n_estimators=10, random_state=42)),
        ('lgrw',  LogisticRegression(penalty = 'l2',random_state= 42,max_iter = 500,solver = 'sag')),
        ('svcw',  LinearSVC(penalty = 'l2',random_state= 42,max_iter = 5000))
                 ]
    
    clf1 = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())        
    
    clf        = clf1.fit(train_features, train_labels)
    #importance = clf.feature_importances_
    print("\n", " Stacking Classifier \n") 

    predictions = clf.predict(train_features)
    conf_mat = confusion_matrix(train_labels, predictions)
    print(conf_mat)
    print(classification_report(train_labels, predictions))
    
    predictions2 = clf.predict(validate_features)
    conf_mat = confusion_matrix(validate_labels, predictions2)
    print(conf_mat)
    print(classification_report(validate_labels,predictions2))
    
    len_train     = len(train_features)
    y_pred_train  = clf.predict(train_features)
    tc_pred_train = np.where(y_pred_train != shift(y_pred_train, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_train     = np.where(y_pred_train == 1, train_future_ret - tc_pred_train,np.where(y_pred_train == -1,-train_future_ret - tc_pred_train,0))
    
    
    print("total Return ",np.mean(ret_train)*100*252,"%")
    print("total tc ",sum(tc_pred_train)*100,"%")
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(252)
    print("sharpe_train",sharpe_train )
    
    len_valid    = len(validate_features)
    y_pred_valid = clf.predict(validate_features)
    tc_pred_valid = np.where(y_pred_valid != shift(y_pred_valid, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_valid    = np.where(y_pred_valid == 1, valid_future_ret - tc_pred_valid,np.where(y_pred_valid == -1, -valid_future_ret - tc_pred_valid,0))
    
    print("total Return ",np.mean(ret_valid)*100*252,"%")
    print("total tc ",sum(tc_pred_valid)*100,"%")
    sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(252)
    print("sharpe_valid",sharpe_valid)
    
    
    
    return clf


def XGBoostRegress(train_features,train_labels,validate_features,validate_labels,train_future_ret,valid_future_ret,features):

    wts2  = np.abs(train_future_ret)*100
    xgbc1 = XGBClassifier(n_estimators = 25,max_depth = 1,eta = 0.1,
                          gamma = 0.0,subsample = 0.6 ,weights = wts2)
    
    xgbc  = xgbc1.fit(train_features, train_labels)
    
    importance = xgbc.feature_importances_
    print("\n", " XGBoost Classifier") 
    
    predictions = xgbc.predict(train_features)
    conf_mat = confusion_matrix(train_labels, predictions)
    print(conf_mat)
    print(classification_report(train_labels, predictions))
    
    predictions2 = xgbc.predict(validate_features)
    conf_mat = confusion_matrix(validate_labels, predictions2)
    print(conf_mat)
    print(classification_report(validate_labels,predictions2))
    
    len_train     = len(train_features)
    y_pred_train  = xgbc.predict(train_features)
    tc_pred_train = np.where(y_pred_train != shift(y_pred_train, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_train     = np.where(y_pred_train == 1, train_future_ret - tc_pred_train,np.where(y_pred_train == -1,-train_future_ret - tc_pred_train,0))
    
    
    print("total Return ",np.mean(ret_train)*100*252,"%")
    print("total tc ",sum(tc_pred_train)*100,"%")
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(252)
    print("sharpe_train",sharpe_train )
    
    len_valid    = len(validate_features)
    y_pred_valid = xgbc.predict(validate_features)
    tc_pred_valid = np.where(y_pred_valid != shift(y_pred_valid, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_valid    = np.where(y_pred_valid == 1, valid_future_ret - tc_pred_valid,np.where(y_pred_valid == -1, -valid_future_ret - tc_pred_valid,0))
    
    print("total Return ",np.mean(ret_valid)*100*252,"%")
    print("total tc ",sum(tc_pred_valid)*100,"%")
    sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(252)
    print("sharpe_valid",sharpe_valid)
    
    
    
    importances = xgbc.feature_importances_
    indices = np.argsort(importances)[(len(importances) - 10):len(importances)]
    
    plt.figure(figsize=(10,10)) 
    plt.title('Adaboost Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
        
    return xgbc,y_pred_train,y_pred_valid


def RandomForestregress(train_features,train_labels,validate_features,validate_labels,train_future_ret,valid_future_ret,features ,ret_thresh):
    
    clf1       = RandomForestRegressor(n_estimators= 50 ,learning_rate= 0.50)#0.5
    clf        = clf1.fit(train_features, train_labels)
    print("\n", " AdaBoost Regressor \n") 

    predictions = clf.predict(train_features)
    predictions2 = clf.predict(validate_features)
    print("r2 train ",r2_score(train_labels, predictions))
    print("r2 valid ",r2_score(validate_labels,predictions2))
    
    thresh = ret_thresh
    
    len_train    = len(train_features)
    y_pred_train = clf.predict(train_features)
    tc_pred_train= np.where(np.sign(y_pred_train) != shift(np.sign(y_pred_train), 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)

    ret_train  = np.where(y_pred_train >thresh, train_future_ret - tc_pred_train,np.where(y_pred_train < -thresh, -train_future_ret - tc_pred_train,0))
    
    print("total Return ",np.mean(ret_train)*100*252,"%")
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(252)
    print("sharpe_train",sharpe_train )
    
    len_valid    = len(validate_features)
    y_pred_valid = clf.predict(validate_features)
    tc_pred_valid= np.where(np.sign(y_pred_valid) != shift(np.sign(y_pred_valid), 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_valid    = np.where(y_pred_valid >thresh, valid_future_ret - tc_pred_valid,np.where(y_pred_valid < -thresh, -valid_future_ret - tc_pred_valid,0))
    
    print("total Return ",np.mean(ret_valid)*100*252,"%")
    sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(252)
    print("sharpe_valid",sharpe_valid)
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(25,25)) 
    plt.title('Adaboost Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    return clf
    

def Adaboostregress(train_features,train_labels,validate_features,validate_labels,train_future_ret,valid_future_ret,features ,ret_thresh):
    
    base_model = DecisionTreeRegressor(max_depth=2)#min_samples_split = 0.10,min_samples_leaf = 0.10)
    clf1       = AdaBoostRegressor(base_model,n_estimators= 50 ,learning_rate= 0.50)#0.5
    clf        = clf1.fit(train_features, train_labels)
    print("\n", " AdaBoost Regressor \n") 

    predictions = clf.predict(train_features)
    predictions2 = clf.predict(validate_features)
    print("r2 train ",r2_score(train_labels, predictions))
    print("r2 valid ",r2_score(validate_labels,predictions2))
    
    thresh = ret_thresh
    
    len_train    = len(train_features)
    y_pred_train = clf.predict(train_features)
    tc_pred_train= np.where(np.sign(y_pred_train) != shift(np.sign(y_pred_train), 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)

    ret_train  = np.where(y_pred_train >thresh, train_future_ret - tc_pred_train,np.where(y_pred_train < -thresh, -train_future_ret - tc_pred_train,0))
    
    print("total Return ",np.mean(ret_train)*100*252,"%")
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(252)
    print("sharpe_train",sharpe_train )
    
    len_valid    = len(validate_features)
    y_pred_valid = clf.predict(validate_features)
    tc_pred_valid= np.where(np.sign(y_pred_valid) != shift(np.sign(y_pred_valid), 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_valid    = np.where(y_pred_valid >thresh, valid_future_ret - tc_pred_valid,np.where(y_pred_valid < -thresh, -valid_future_ret - tc_pred_valid,0))
    
    print("total Return ",np.mean(ret_valid)*100*252,"%")
    sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(252)
    print("sharpe_valid",sharpe_valid)
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(25,25)) 
    plt.title('Adaboost Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    return clf


def test_performance(clf ,test_features,test_labels,test_future_ret,features):

    predictions = clf.predict(test_features)
    conf_mat = confusion_matrix(test_labels, predictions)
    print(conf_mat)
    print(classification_report(test_labels, predictions))
    
    
    len_test    = len(test_features)
    y_pred_test = clf.predict(test_features)
    tc_pred_test= np.where(y_pred_test != shift(y_pred_test, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_test    = np.where(y_pred_test == 1, test_future_ret - tc_pred_test,np.where(y_pred_test == -1, -test_future_ret - tc_pred_test,0))
    cum_ret_test = np.cumsum(ret_test)
    
    print("total Return ",np.mean(ret_test)*100*252,"%")
    print("total tc ",sum(tc_pred_test)*100,"%")
    sharpe_test = ret_test.mean()/ret_test.std()*np.sqrt(252)
    print("sharpe_test",sharpe_test)
    
    
    return (y_pred_test)

    
def test_performance_regress(clf ,test_features,test_labels,test_future_ret,features,ret_thresh):

    predictions = clf.predict(test_features)
    print("r2 train ",r2_score(test_labels, predictions))
    
    thresh = ret_thresh
    
    len_test    = len(test_features)
    y_pred_test = clf.predict(test_features)
    tc_pred_test= np.where(np.sign(y_pred_test) != shift(np.sign(y_pred_test), 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
    ret_test    = np.where(y_pred_test >thresh, test_future_ret - tc_pred_test,np.where(y_pred_test < -thresh, -test_future_ret - tc_pred_test,0))
    
    print("total Return ",np.mean(ret_test)*100*252,"%")
    print("total tc ",sum(tc_pred_test)*100,"%") 
    sharpe_test = ret_test.mean()/ret_test.std()*np.sqrt(252)
    print("sharpe_test ",sharpe_test )
    