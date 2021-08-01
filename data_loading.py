
import os
import pandas as pd
import constants as constx
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pyreadr
import json


#import numpy as np


def base_data_loader():
    
    data_folder2 = constx.data_folder
    
    os.chdir(data_folder2)
    total_data               = pd.read_csv("corp_action_px_daily2"+ ".csv")
    
    total_data['DATE'] = pd.to_datetime(total_data['DATE'])
    
    
    return (total_data)

def base_data_loader2():
    
    data_folder2 = constx.base_folder
    os.chdir(data_folder2)
    
    #total_data               = pd.read_excel('corp_ann_pdfs_temp.xlsx')
    total_data               = pd.read_excel('corp_ann_pdfs.xlsx')
        
    
    return (total_data)

def stopwords_loader():
    
    data_folder2 = constx.base_folder
    os.chdir(data_folder2)
    
    f  = open('stopwords.txt')
    all_words = f.read()
    all_words = all_words.split('\n')
    
    return all_words

def groupby_date_symbol(total_data):
    
    total_datax1 = total_data[{'DATE' ,'SYMBOL' ,'TEXT'}]
    total_datax2 = total_data[{'DATE' ,'SYMBOL' ,'PX1','PX2','PX3','PX5','PX10'}]
    
    total_data2 = total_datax1.groupby(['DATE' ,'SYMBOL']).agg('\n'.join).reset_index()
    total_data3 = total_datax2.groupby(['DATE' ,'SYMBOL']).agg(lambda x : max(x)).reset_index()
            
    total_data3['TEXT'] = total_data2.TEXT    
    total_data3 = total_data3.reset_index(drop = True)
        
    
    return total_data3
    
    
    
    
    
    
    
    
    
    
    


    
    

