import constants as constx

import tensorflow
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Flatten


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import pandas as pd
import numpy as np
from scipy.ndimage.interpolation import shift
import random
import matplotlib.pyplot as plt



def LogisticModel(train_features,train_labels,validate_features,
               validate_labels,train_future_ret,valid_future_ret,features):
    
        model = LogisticRegression(penalty = 'l1' , solver = 'saga',)
                                   #class_weight = {1:0.45 ,-1:0.45,0:0.1} )
        model.fit(train_features, train_labels)
        predictions = model.predict(train_features)
        conf_mat    = confusion_matrix(train_labels, predictions)
        print(conf_mat)
        print(classification_report(train_labels, predictions))
        
        predictions2 = model.predict(validate_features)
        conf_mat     = confusion_matrix(validate_labels, predictions2)
        print(conf_mat)
        print(classification_report(validate_labels,predictions2))
        
        len_train     = len(train_features)
        y_pred_train  = np.reshape(predictions ,(predictions.shape[0],))
        tc_pred_train = np.reshape(np.where(y_pred_train != 0,constx.TRANSACTION_COST_BPS,0),(y_pred_train.shape[0],))
        
        
        ret_train     = np.where(y_pred_train == 1, train_future_ret - tc_pred_train,
                         np.where(y_pred_train == -1, -train_future_ret - tc_pred_train,0))    
        
        print("total Return ",np.mean(ret_train)*100*252,"%")
        print("total tc ",sum(tc_pred_train)*100,"%")
        sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(252)
        print("sharpe_train",sharpe_train )
        
        len_valid     = len(validate_features)
        y_pred_valid  = np.reshape(predictions2 ,(predictions2.shape[0],))
        tc_pred_valid = np.reshape(np.where(y_pred_valid != 0,constx.TRANSACTION_COST_BPS,0),(y_pred_valid.shape[0],))
        
        ret_valid     = np.where(y_pred_valid == 1, valid_future_ret - tc_pred_valid,
                         np.where(y_pred_valid == -1, -valid_future_ret - tc_pred_valid,0))
        
        print("total Return ",np.mean(ret_valid)*100*252,"%")
        print("total tc ",sum(tc_pred_valid)*100,"%")
        sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(252)
        print("sharpe_valid",sharpe_valid)


def NeuralNetwork_model(train_features,train_labels,validate_features,
               validate_labels,train_future_ret,valid_future_ret,features):
    
    model = Sequential()
    # model.add(Embedding(input_dim= train_features.shape[1], 
    #                            output_dim= constx.EMBEDDING_DIM, 
    #                            input_length= train_features.shape[1]))
    
    model.add(Dense(20, activation='relu',input_dim= train_features.shape[1]))
    # model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dense(1, activation='softmax'))
    
    opt  = tensorflow.keras.optimizers.Adam(lr = 1e-4)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    #train_labels2 = pd.DataFrame(to_categorical(train_labels))

    
    print(model.summary())
    
    epochs     = constx.epochs
    batch_size = constx.batch_size
        
    history = model.fit(train_features, train_labels, epochs=epochs, 
                        batch_size=batch_size)
    
    loss, accuracy = model.evaluate(train_features, train_labels)
    print("accuracy ")

    predictions = model.predict(train_features)
    conf_mat    = confusion_matrix(train_labels, predictions)
    print(conf_mat)
    print(classification_report(train_labels, predictions))
    
    predictions2 = model.predict(validate_features)
    conf_mat     = confusion_matrix(validate_labels, predictions2)
    print(conf_mat)
    print(classification_report(validate_labels,predictions2))
    
    len_train     = len(train_features)
    y_pred_train  = np.reshape(predictions ,(predictions.shape[0],))
    tc_pred_train = np.reshape(np.where(y_pred_train != 0,constx.TRANSACTION_COST_BPS,0),(y_pred_train.shape[0],))
    
    
    ret_train     = np.where(y_pred_train == 1, train_future_ret - tc_pred_train,
                             np.where(y_pred_train == -1, -train_future_ret - tc_pred_train,0))    
    
    print("total Return ",np.mean(ret_train)*100*252,"%")
    print("total tc ",sum(tc_pred_train)*100,"%")
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(252)
    print("sharpe_train",sharpe_train )
    
    len_valid     = len(validate_features)
    y_pred_valid  = np.reshape(predictions2 ,(predictions2.shape[0],))
    tc_pred_valid = np.reshape(np.where(y_pred_valid != 0,constx.TRANSACTION_COST_BPS,0),(y_pred_valid.shape[0],))

    ret_valid     = np.where(y_pred_valid == 1, valid_future_ret - tc_pred_valid,
                             np.where(y_pred_valid == -1, -valid_future_ret - tc_pred_valid,0))
    
    print("total Return ",np.mean(ret_valid)*100*252,"%")
    print("total tc ",sum(tc_pred_valid)*100,"%")
    sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(252)
    print("sharpe_valid",sharpe_valid)

    return model
    

def lstm_model(train_features,train_labels,validate_features,
               validate_labels,train_future_ret,valid_future_ret,features):
    
    model = Sequential()
    model.add(Embedding(constx.MAX_NUM_WORDS, constx.EMBEDDING_DIM, input_length=train_features.shape[1]))
    #model.add(LSTM(100, dropout=0.5))
    model.add(Dense(10, activation='relu'))    
    model.add(Dense(10, activation='relu'))    
    model.add(Dense(1, activation='softmax'))
        
    opt = tensorflow.keras.optimizers.Adam(learning_rate = 0.1)
    model.compile(loss='binary_crossentropy', optimizer= opt,
                  metrics=['accuracy'])
    print(model.summary())
    
    epochs     = constx.epochs
    batch_size = constx.batch_size
    
    history = model.fit(train_features, train_labels, epochs=epochs, 
                        batch_size=batch_size,validation_split= constx.VALIDATION_SPLIT)#,
                        #class_weight = {0:0.60 ,1:0.40})
    
    predictions = model.predict(train_features)
    conf_mat    = confusion_matrix(train_labels, predictions)
    print(conf_mat)
    print(classification_report(train_labels, predictions))
    
    predictions2 = model.predict(validate_features)
    conf_mat     = confusion_matrix(validate_labels, predictions2)
    print(conf_mat)
    print(classification_report(validate_labels,predictions2))
    
    len_train     = len(train_features)
    y_pred_train  = np.reshape(predictions ,(predictions.shape[0],))
    tc_pred_train = np.reshape(np.where(y_pred_train != 0,constx.TRANSACTION_COST_BPS,0),(y_pred_train.shape[0],))
    
    
    ret_train     = np.where(y_pred_train == 1, train_future_ret - tc_pred_train,
                             np.where(y_pred_train == -1, -train_future_ret - tc_pred_train,0))    
    
    print("total Return ",np.mean(ret_train)*100*252,"%")
    print("total tc ",sum(tc_pred_train)*100,"%")
    sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(252)
    print("sharpe_train",sharpe_train )
    
    len_valid     = len(validate_features)
    y_pred_valid  = np.reshape(predictions2 ,(predictions2.shape[0],))
    tc_pred_valid = np.reshape(np.where(y_pred_valid != 0,constx.TRANSACTION_COST_BPS,0),(y_pred_valid.shape[0],))

    ret_valid     = np.where(y_pred_valid == 1, valid_future_ret - tc_pred_valid,
                             np.where(y_pred_valid == -1, -valid_future_ret - tc_pred_valid,0))
    
    print("total Return ",np.mean(ret_valid)*100*252,"%")
    print("total tc ",sum(tc_pred_valid)*100,"%")
    sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(252)
    print("sharpe_valid",sharpe_valid)

    return model
    

def SupportVM(train_features,train_labels,validate_features,
              validate_labels,train_future_ret,valid_future_ret,features):
    
        model =  SVC(kernel ='rbf',C = 2.50 ,)#class_weight = {1:0.40 ,-1:0.40,0:0.2})   
        
        
        model.fit(train_features, train_labels)
        predictions = model.predict(train_features)
        conf_mat    = confusion_matrix(train_labels, predictions)
        print(conf_mat)
        print(classification_report(train_labels, predictions))
        
        predictions2 = model.predict(validate_features)
        conf_mat     = confusion_matrix(validate_labels, predictions2)
        print(conf_mat)
        print(classification_report(validate_labels,predictions2))
        
        len_train     = len(train_features)
        y_pred_train  = np.reshape(predictions ,(predictions.shape[0],))
        tc_pred_train = np.reshape(np.where(y_pred_train != 0,constx.TRANSACTION_COST_BPS,0),(y_pred_train.shape[0],))
        
        
        ret_train     = np.where(y_pred_train == 1, train_future_ret - tc_pred_train,
                         np.where(y_pred_train == -1, -train_future_ret - tc_pred_train,0))    
        
        print("total Return ",np.mean(ret_train)*100*252,"%")
        print("total tc ",sum(tc_pred_train)*100,"%")
        sharpe_train = ret_train.mean()/ret_train.std()*np.sqrt(252)
        print("sharpe_train",sharpe_train )
        
        len_valid     = len(validate_features)
        y_pred_valid  = np.reshape(predictions2 ,(predictions2.shape[0],))
        tc_pred_valid = np.reshape(np.where(y_pred_valid != 0,constx.TRANSACTION_COST_BPS,0),(y_pred_valid.shape[0],))
        
        ret_valid     = np.where(y_pred_valid == 1, valid_future_ret - tc_pred_valid,
                         np.where(y_pred_valid == -1, -valid_future_ret - tc_pred_valid,0))
        
        print("total Return ",np.mean(ret_valid)*100*252,"%")
        print("total tc ",sum(tc_pred_valid)*100,"%")
        sharpe_valid = ret_valid.mean()/ret_valid.std()*np.sqrt(252)
        print("sharpe_valid",sharpe_valid)

    

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
    
    print("total Return ",np.mean(ret_test)*100*365*48,"%")
    print("total tc ",sum(tc_pred_test)*100,"%")
    sharpe_test = ret_test.mean()/ret_test.std()*np.sqrt(365*48)
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
    
    print("total Return ",np.mean(ret_test)*100*365*48,"%")
    print("total tc ",sum(tc_pred_test)*100,"%") 
    sharpe_test = ret_test.mean()/ret_test.std()*np.sqrt(365*48)
    print("sharpe_test ",sharpe_test )
    