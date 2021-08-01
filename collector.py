base_folder = "C://projects//DATA//SENTIMENT//CRYPTO//python"
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

dfx = flm.load_all_feature_types2(total_data)

dfx            = dfx.replace([np.inf, -np.inf], np.nan)
dfx            = dfx.dropna()
dfx['details'] = dfx['details'].apply(mlm.clean_text2)

new_features = mlm.tokenize_values(dfx)
dfx          = pd.DataFrame(new_features)
dfx          = pd.concat([dfx.reset_index(drop=True), label_data], axis=1)

dfx            = dfx.replace([np.inf, -np.inf], np.nan)
dfx            = dfx.dropna()
dfx            = dfx.reset_index(drop = True)

