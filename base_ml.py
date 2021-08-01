
base_folder = "C://projects//DATA//downloaders//NSE_CORPORATE_ANNOUNCEMENTS//python"
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
import cross_valid as cv

import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(constx.seedx)

total_data    = data_loading_module.base_data_loader2()
#label_data    = flm.label_multiple(total_data ,0.05) 
#label_data    = flm.label_binary(total_data)
label_data    = flm.label_binary2(total_data)

dfx2 = flm.load_all_feature_types2(total_data)

dfx2            = dfx2.replace([np.inf, -np.inf], np.nan)
dfx2            = dfx2.dropna()
dfx2['details'] = dfx2['details'].apply(mlm.clean_text2)

new_features = mlm.tokenize_values(dfx2)

dfx          = new_features
dfx          = pd.concat([dfx.reset_index(drop=True), label_data], axis=1)

dfx            = dfx.replace([np.inf, -np.inf], np.nan)
dfx            = dfx.dropna()
dfx            = dfx.reset_index(drop = True)

#train ,validate ,test = tts.train_validate_test_split_sequential(dfx ,constx.train_percent ,constx.valid_percent)
train ,validate ,test = tts.train_validate_test_split_random(dfx ,constx.train_percent ,constx.valid_percent ,seed = constx.seedx)

features = dfx.columns.tolist()
features = [e for e in features if e not in ('label', 'future_ret')]

train_features    = train[features]
validate_features = validate[features]
test_features     = test[features]

train_labels     = train['label']
validate_labels  = validate['label']
test_labels      = test['label']

train_future_ret = train['future_ret']
valid_future_ret = validate['future_ret']
test_future_ret  = test['future_ret']

train_feature_list = list(train_features.columns)

model = mlm.lstm_model(train_features ,train_labels ,validate_features ,
                       validate_labels ,train_future_ret,valid_future_ret,train_feature_list)


print("\n",features)

