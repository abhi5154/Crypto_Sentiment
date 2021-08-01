
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
import csv
import time
import pandas as pd
import os

data_folder  = "C://projects//DATA//downloaders//NSE_CORPORATE_ANNOUNCEMENTS//python"
os.chdir(data_folder)
nrow = 40

moneycontrol = []

path   = r'C:/Users/Abhishek/Downloads/Compressed/chromedriver_win32/chromedriver.exe'
driver = webdriver.Chrome(executable_path= path)
url    = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"

driver.get(url)

name_stock = 'RELIANCE'

xpath00 =  '/html/body/div[7]/div[1]/div/section/div/div/div/div/div/div[1]/div[1]/div[1]/div/span/input[1]'
ll00 = driver.find_element_by_class_name('form-control with_radius companyVal typeahead companyAutoComplete tt-input active')
#ll00 = driver.find_element_by_xpath(xpath00)
ll00.send_keys(name_stock)



outx1 = ll00.text
outx2 = ll01.text
outx3 = ll02.text
outx4 = ll03.get_attribute('href')


driver.get(outx4)

str_total = ""
elem = driver.find_elements_by_css_selector('p')
for el in elem:
    str_total = str_total+ '\n' + el.text
    #print(el.text)


driver.get(url)



moneycontrol.append([outx1 ,outx2,outx3,str_total])


for i in range(1,3):
        
    urlx = "https://www.moneycontrol.com/news/business/stocks/page-" + str(i) + "/"
    driver.get(urlx)
    #driver.implicitly_wait(5)
    time.sleep(2)
    print(i)

    
    for j in range(0,33):
        
        try:
            
            
            xpath00 =  '/html/body/section/div/ul/li[' + str(j) + ']/span'
            xpath01 =  '/html/body/section/div/ul/li[' + str(j) + ']/h2/a'
            xpath02 =  '/html/body/section/div/ul/li[' + str(j) + ']/p[1]'
            xpath03 =  '/html/body/section/div/ul/li[' + str(j) + ']/a'
            
            ll00 = driver.find_element_by_xpath(xpath00)
            ll01 = driver.find_element_by_xpath(xpath01)
            ll02 = driver.find_element_by_xpath(xpath02)
            ll03 = driver.find_element_by_xpath(xpath03)
            
            outx1 = ll00.text
            outx2 = ll01.text
            outx3 = ll02.text
            outx4 = ll03.get_attribute('href')
            
            
            driver.get(outx4)
            
            str_total = ""
            elem = driver.find_elements_by_css_selector('p')
            for el in elem:
                str_total = str_total+ '\n' + el.text
                #print(el.text)
            
            moneycontrol.append([outx1 ,outx2,outx3,str_total])
            #driver.implicitly_wait(5)
            time.sleep(2)

            
            driver.get(urlx)
            #driver.implicitly_wait(5)
            time.sleep(2)

            
            str0 = "window.scrollTo(0," + str(j*30 + 10) + ");"
            driver.execute_script(str0)
                        
        except Exception as e:
            print(e)
            #driver.implicitly_wait(5)
            time.sleep(2)

            #print (str(i) + " " + str(j))
    
    #driver.implicitly_wait(5)
    time.sleep(2)


            



money2 = pd.DataFrame(moneycontrol ,columns = ['DATE','TITLE','DESCRIPTION','BODY'])
money2.to_csv('moneycontrol_news_stocks.csv' ,index = False)




