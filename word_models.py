
import numpy as np
import pandas as pd

import data_loading as dlm
import constants as constx


import spacy  # For preprocessing

from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess
from gensim.matutils import unitvec
from gensim.models import Word2Vec, word2vec
from gensim.models import KeyedVectors
import gensim.downloader as api


import re
from nltk import word_tokenize
from nltk import sent_tokenize
import nltk
from nltk.corpus import stopwords


from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import fasttext

#nltk.download('punkt')

english_words = set(nltk.corpus.words.words())
STOPWORDS = set(stopwords.words('english'))

#nltk.download('stopwords')




def clean_text0(text):
    """
        text: a string
        
        return: modified initial string
    """
    text  = text.replace('\n', ' ')
    text  = text.lower() # lowercase text

    text = re.sub(" +"," ", text)                  ##extra white spaces
    text = ' '.join( [w for w in text.split() if len(w)>1] )

    text = ''.join([i for i in text if not i.isdigit()])

    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;+#]')
    BAD_SYMBOLS_RE      = re.compile('[^0-9a-z #+_]')

    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub(' ', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 

    text = re.sub(" +"," ", text)                  ##extra white spaces
    text = ' '.join( [w for w in text.split() if len(w)>2] )
    text = re.sub(" +"," ", text)                  ##extra white spaces
    
    return text


def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE      = re.compile('[^0-9a-z #+_]')
    
    text = text.lower() # lowercase text
    
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    
    return text

def clean_text2(text):
    """
        text: a string
        
        return: modified initial string
    """
    
    stopwords2 = dlm.stopwords_loader()
    
    text  = text.lower() # lowercase text
    
    text = re.sub(r'^www.*.com', '', text, flags=re.MULTILINE)
    text = text.encode('ascii', 'ignore').decode() ##remove Unicode characters
    
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;+#]')
    BAD_SYMBOLS_RE      = re.compile('[^0-9a-z #+_]')

    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub(' ', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 

    text = ''.join([i for i in text if not i.isdigit()])
    
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    
    text = re.sub(" +"," ", text)                  ##extra white spaces
    
    querywords = text.split()    
    text2  = [word for word in querywords if word not in stopwords2]
    text   = ' '.join(text2)

    text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in english_words or not w.isalpha())

    text = re.sub(" +"," ", text)                  ##extra white spaces

    
    return text

def clean_text3(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.replace('\n', ' ')
    text  = text.lower() # lowercase text

    text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in english_words or not w.isalpha())
    text = re.sub(" +"," ", text)                  ##extra white spaces
    text = ' '.join( [w for w in text.split() if len(w)>1] )

    text = ''.join([i for i in text if not i.isdigit()])

    # stop_words = set(stopwords.words('english'))
    # text =  ' '.join([word for word in text.split() if word not in STOPWORDS])    


    #text = ' '.join( [w for w in text.split() if len(w)>1] )
    text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in english_words or not w.isalpha())

    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;+#]')
    BAD_SYMBOLS_RE      = re.compile('[^0-9a-z #+_]')

    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub(' ', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 

    text = re.sub(" +"," ", text)                  ##extra white spaces
    text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in english_words or not w.isalpha())
    text = re.sub(" +"," ", text)                  ##extra white spaces
    text =  ' '.join([word for word in text.split() if word not in STOPWORDS])    
    text = ' '.join( [w for w in text.split() if len(w)>2] )

    stopwords2 = dlm.stopwords_loader()        
    querywords = text.split()    
    text2  = [word for word in querywords if word not in stopwords2]
    text   = ' '.join(text2)

    
    return text


def count_tokenize_values(dfx):   
    
    # The maximum number of words to be used. (most frequent)
    
    # Max number of words in each complaint.
    
    # This is fixed.
    
    vectorizer = CountVectorizer(max_features = constx.MAX_NUM_WORDS)
    vectorizer.fit(dfx['details'].values)
        
    X = vectorizer.transform(dfx['details'].values)
    dfx          = pd.DataFrame(X.toarray())

    return (dfx)

def tokenize_values(dfx):   
        
    
    tokenizer = Tokenizer(num_words= constx.MAX_NUM_WORDS)
    tokenizer.fit_on_texts(dfx['details'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    X = tokenizer.texts_to_sequences(dfx['details'].values)
    X = pad_sequences(X, maxlen= constx.MAX_SEQUENCE_LENGTH)
    
    dfx          = pd.DataFrame(X)
    
    return (dfx)

def tokenize_word2vec(dfx2):
    
    def tokenize(doc):        
        wordList = re.sub("[^\w]", " ",  doc).split()
        return wordList
    
    X  = dfx2['details'].apply(tokenize)

    def word_averaging(w2v_model, words):
        all_words, mean = set(), []
        
        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in w2v_model.wv.key_to_index:
                mean.append(w2v_model.wv[w2v_model.wv.key_to_index[word]])
    
        if not mean:
            return np.zeros(w2v_model.layer1_size,)
    
        mean = unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

    
    def  word_averaging_list(wv, text_list):
        return np.vstack([word_averaging(wv, review) for review in text_list ])


    w2v_model = Word2Vec(X ,min_count= 1,vector_size=300)
    
    X_train_word_average = word_averaging_list(w2v_model,X)
    
    dfx          = pd.DataFrame(X_train_word_average)
    
    return dfx

def tokenize_word2vec_google(dfx2):
    
    def tokenize(doc):        
        wordList = re.sub("[^\w]", " ",  doc).split()
        return wordList
    
    X  = dfx2['details'].apply(tokenize)

    def word_averaging(w2v_model, words):
        all_words, mean = set(), []
        
        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in w2v_model.wv.key_to_index:
                mean.append(w2v_model.wv[w2v_model.wv.key_to_index[word]])
    
        if not mean:
            return np.zeros(w2v_model.layer1_size,)
    
        mean = unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

    
    def  word_averaging_list(wv, text_list):
        return np.vstack([word_averaging(wv, review) for review in text_list ])

    w2v_model = api.load("glove-twitter-25")

    #w2v_model = KeyedVectors.load_word2vec_format("glove.6B.300d.txt", binary=False)
    
    X_train_word_average = word_averaging_list(w2v_model,X)
    
    dfx          = pd.DataFrame(X_train_word_average)
    
    return dfx


def fsttext_clean(dfx2,label_data):
    
    labels1 = label_data['label'].values
    labels1[labels1==-1] = 2    
    labels2 = pd.DataFrame(["__labels__" + str(s) for s in labels1])
    labels2.columns = ["label"]
    
    dfx2_text = dfx2.details

    label3 = labels2.label.str.cat(dfx2_text, sep=' ')    
    
    return label3


def fasttext_model(train_features,train_labels,validate_features,validate_labels,train_future_ret,valid_future_ret,features):
    return 0
    
    
















