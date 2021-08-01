
base_folder = "C://projects//DATA//downloaders//SENTIMENT//python"
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir(base_folder)

import numpy as np
import constants as constx
import feature_label_maker as flm
import data_loading as data_loading_module
import train_test_splits as tts
import ml_models as mlm
import ml_models2 as mlm2
import word_models as wrdm
import cross_valid as cv
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
import pandas as pd
import fasttext


np.random.seed(constx.seedx)

total_data    = data_loading_module.base_data_loader2()
total_data2   = data_loading_module.groupby_date_symbol(total_data)

label_data    = flm.label_multiple(total_data2 ,0.02) 
#label_data    = flm.label_binary(total_data2)
#label_data    = flm.label_binary2(total_data2)

dfx3 = flm.load_all_feature_types2(total_data2)

dfx3            = dfx3.replace([np.inf, -np.inf], np.nan)
dfx3            = dfx3.dropna()

dfx2  = pd.DataFrame()
dfx2['details'] = dfx3['details'].apply(wrdm.clean_text0)


dfx              = wrdm.fsttext_clean(dfx2,label_data)

lngth = [len(dfx[i]) >13 for i in dfx.index]
dfxm = dfx[lngth]

embedding_size = 60
window_size = 40
min_word = 5
down_sampling = 1e-2



ft_model = FastText(dfxm,size=embedding_size,window=window_size,
                      min_count=min_word,sample=down_sampling,sg=1,iter=100)



























