
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def update_prices(datess1 ,symbol_name):
    
    PX1 = np.NAN;PX2 = np.NAN;PX3 = np.NAN;PX5 = np.NAN;PX10 = np.NAN;
    
    tod1 = datess1 + timedelta(days=1)
    tod1 = next(dts for dts in px_data.index if dts >= tod1)     
    PX1  = px_data.loc[px_data.index == tod1,symbol_name].values[0]
    
    tod2 = datess1 + timedelta(days=2)
    tod2 = next(dts for dts in px_data.index if dts >= tod2)     
    PX2  = px_data.loc[px_data.index == tod2,symbol_name].values[0]
    
    tod3 = datess1 + timedelta(days=3)
    tod3 = next(dts for dts in px_data.index if dts >= tod3)     
    PX3  = px_data.loc[px_data.index == tod3,symbol_name].values[0]
    
    tod5 = datess1 + timedelta(days=5)
    tod5 = next(dts for dts in px_data.index if dts >= tod5)     
    PX5  = px_data.loc[px_data.index == tod5,symbol_name].values[0]
    
    tod10 = datess1 + timedelta(days=10)
    tod10 = next(dts for dts in px_data.index if dts >= tod10)     
    PX10  = px_data.loc[px_data.index == tod10,symbol_name].values[0]
    
    vec = [PX1 ,PX2 ,PX3 ,PX5,PX10]
    return (vec)
   

links_folder     = 'C:\\projects\\DATA\\downloaders\\NSE_CORPORATE_ANNOUNCEMENTS\\DATA\\CORP\\LINKS'
links_pdf_folder = 'C:\\projects\\DATA\\downloaders\\NSE_CORPORATE_ANNOUNCEMENTS\\DATA\\CORP\\LINKS\\LINK PDFS'
links_txt_folder = 'C:\\projects\\DATA\\downloaders\\NSE_CORPORATE_ANNOUNCEMENTS\\DATA\\CORP\\LINKS\\LINK TXTS'

data_folder      = 'C:\\projects\\SENTIMENT1\\DATA'
base_folder = "C://projects//DATA//downloaders//NSE_CORPORATE_ANNOUNCEMENTS//python"

os.chdir(data_folder)
nifty50       = pd.read_csv('nifty50.csv' )
px_data       = pd.read_csv('price_all_investing.csv')
px_data       = px_data.set_index('DATETIME' ,drop = True)
px_data.index =  pd.to_datetime(px_data.index).date


ALL_DATA   = pd.DataFrame(columns = ['DATE' ,'SYMBOL','TEXT' ,'PX1' ,'PX2','PX3', 'PX5' ,'PX10'])

for ii in np.arange(0,len(nifty50.index)):    
    
    stock_name = nifty50.loc[ii ,'Symbol']
    print(stock_name)
    os.chdir(links_txt_folder)
    os.chdir(stock_name)
  
    list_files = os.listdir(os.getcwd())
  
    for jj in np.arange(0,len(list_files)):
        
        #print(jj)
        file1 = list_files[jj]
        f     = open(file1, "r", errors="ignore")
        txt0  = f.read()
        f.close()
                
        if(len(txt0) <5):
            next
      
        else:
            try:                
                symbol_name      = stock_name
                dttime           = datetime.strptime(file1[0:len(file1)-4] , "%m%d%Y%H%M%S")
                datess1          = dttime.date()
                prices           = update_prices(datess1 ,symbol_name)
                
                new_data    = [datess1,symbol_name,txt0,prices[0],prices[1],prices[2],prices[3],prices[4]]
                ALL_DATA.loc[len(ALL_DATA)] = new_data
                
            except:
                next
                
                

os.chdir(base_folder)      
ALL_DATA.to_excel('corp_ann_pdfs.xlsx' ,index = False)

#hy_data = pd.read_excel('corp_ann_pdfs.xlsx')


