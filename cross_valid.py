
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree     import DecisionTreeClassifier
from xgboost import XGBClassifier


from sklearn.metrics import make_scorer
import numpy as np
import constants as constx
from scipy.ndimage.interpolation import shift



def sharpe_ratio_scorer(y_true, y_predict ,train_future_ret):
    
    len_train     = len(y_true)
    tc_pred_train = np.where(y_predict != shift(y_predict, 1, cval=np.NaN),constx.TRANSACTION_COST_BPS,0)
        
    ret_train     = np.where(y_predict == 1, train_future_ret - tc_pred_train,
                             np.where(y_predict == -1,-train_future_ret - tc_pred_train,0))
    
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(365*48)
    
    return sharpe_train


#sharpe_ratio_scorer = make_scorer(sharpe_ratio_scorer, greater_is_better= True)

def CV_NeuralNetwork(train_features ,train_labels ,train_future_ret):
    
    mlp_gs = MLPClassifier(max_iter=1000)

    parameter_space = {
        'hidden_layer_sizes' : [(20,20,20),(10,10,10),(30,30),(5,5,5,5)],
        'activation': [ 'relu','tanh'],
        'solver': ['sgd'],
        'alpha': [0.0001,0.001,0.01,0.1],
        'learning_rate': ['adaptive']
    }
    
    clf0 = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
    clf0.fit(train_features, train_labels) # X is train samples and y is the corresponding labels

    return clf0

def CV_BagClass(train_features ,train_labels ,train_future_ret):
    
    parameter_space = {
        'base_estimator__hidden_layer_sizes' : [(3,3),(2,2),(5,5)],
        'base_estimator__activation'         : [ 'relu'],
        'base_estimator__solver'             : ['sgd'],
        'base_estimator__alpha'              : [0.001,0.01],
        'base_estimator__learning_rate'      : ['adaptive'],
        'base_estimator__max_iter'           : [500],
        'base_estimator__tol'                : [0.0001,0.001],
        'n_estimators'                       : [20]
    }
    
    clf0 = GridSearchCV(BaggingClassifier(MLPClassifier()), parameter_space, n_jobs=-1, cv= 3)
    clf0.fit(train_features, train_labels) # X is train samples and y is the corresponding labels

    return clf0

def CV_AdaBoost(train_features ,train_labels ,train_future_ret):
        
    parameter_space = {
        'base_estimator__max_depth'         : [1,2],
        'base_estimator__min_samples_split' : [0.05,0.1,0.15,0.2],
        'base_estimator__min_samples_leaf'  : [0.05,0.1,0.15,0.2],
        'base_estimator__class_weight'      : ['balanced'],
        'n_estimators'                      : [50],
        'learning_rate'                     : [0.25,0.5,0.75,1]        
    }
    
    clf0 = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier()), parameter_space, n_jobs=-1, cv= 5)
    clf0.fit(train_features, train_labels) # X is train samples and y is the corresponding labels

    return clf0

def CV_XGBoost(train_features ,train_labels ,train_future_ret):
    
    parameter_space = {        
        'eta'           : [0.1,0.3,0.5],
        'gamma'         : [0,0.5,2.0],
        'max_depth'     : [1,2],
        'subsample'     : [0.4,0.6,1.0],
        'n_estimators'  : [25,50]
        
    }
    
    clf0 = GridSearchCV(XGBClassifier(), parameter_space, n_jobs=-1, cv=5)
    clf0.fit(train_features, train_labels) # X is train samples and y is the corresponding labels

    return clf0

def CV_randomforest(train_features ,train_labels ,train_future_ret):
    
    parameter_space = {        
        'max_depth'         : [1,2,3],
        'n_estimators'      : [20,40,60],
        'min_samples_split' : [0.05 ,0.10 ,0.20, 0.25],
        'min_samples_leaf'  : [0.05 ,0.10 ,0.20, 0.25],
        'max_features'      : [0.33,0.67,1.0],
        #'min_weight_fraction_leaf' : [0.05,0.1,0.2],
        #'min_impurity_decrease' : [0.05,0.1,0.2],
        'class_weight'          : ['balanced', 'balanced_subsample']
                
    }
    
    clf0 = GridSearchCV(RandomForestClassifier(), parameter_space, n_jobs=-1, cv=5)
    clf0.fit(train_features, train_labels) # X is train samples and y is the corresponding labels

    return clf0

