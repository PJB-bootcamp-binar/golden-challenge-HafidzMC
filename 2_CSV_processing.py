import streamlit as st
import re
import pandas as pd
from io import StringIO
import chardet
import sqlite3
from sqlite3 import Connection

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

import nltk
#nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('indonesian')

from nltk.tokenize.treebank import TreebankWordDetokenizer

############################################################

def remove_punct(tweet):
    # tweet = re.sub('\[.*\]', '', str(tweet)).strip()    #remove text in square brackets
    # tweet = tweet.translate(str.maketrans('','',string.punctuation))    #remove punctuation
    # tweet = re.sub('\S*\d\S*\s', '', str(tweet)).strip()   #remove text in square brackets
    # return tweet.strip()
    tweet = re.sub('[^a-zA-Z0-9 ]', ' ', str(tweet))
    tweet = re.sub('[0-9]+', ' ', tweet)
    tweet = re.sub(r'#', '', str(tweet))  
    tweet = re.sub(r'http\S+', ' ', tweet)
    tweet = re.sub(r'\s+|\\n', ' ', tweet)
    tweet = re.sub('RT[\s]+', '', tweet) # Removing RT
    tweet = re.sub('https?:\/\/\S+', '', tweet) # Removing hyperlink
    return tweet

def tokenization(tweet):
    tweet = re.split('\W+', tweet)
    return tweet

def normalisasi(tweet):
    kamusalay_dict = dict(df_uploaded_kamusalay.values)
    pattern = re.compile(r'\b( ' + '|'.join(kamusalay_dict.keys())+r')\b') # Search pola kata (contoh kpn -> kapan)
    content = []
    for kata in tweet:
        filteredSlang = pattern.sub(lambda x: kamusalay_dict[x.group()],kata) # Replace slangword berdasarkan pola review yg telah ditentukan
        content.append(filteredSlang.lower())
    tweet = content
    return tweet

def remove_stopwords(tweet):
    tweet = [word for word in tweet if word not in stopword]
    return tweet

def stemming(tweet):
    tweet = [stemmer.stem(word) for word in tweet]
    return tweet


URI_SQLITE_DB = "pages/data/data_cleansing.db"

def get_connection(path: str):
    return sqlite3.connect(path, check_same_thread=False)

def init_db(conn: Connection):
    conn.execute("""CREATE TABLE IF NOT EXISTS result (RAW_DATA, CLEANED_DATA);""")
    conn.commit()

def get_data(conn: Connection):
    dfr = pd.read_sql("SELECT CLEANED_DATA as CLEANED_SQLITE3_DATA FROM result", con=conn)
    return dfr

def display_data(conn: Connection):
    st.dataframe(get_data(conn))

############################################################

st.write("# Endpoint 2: text cleansing from CSV input")

st.markdown("""fitur ini berfungsi untuk pengolahan kata melalui input file CSV *""")

uploaded_file = st.file_uploader("silahkan upload file CSV yang akan di proses")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding = "ISO-8859-1")
    max_data = st.slider("Max data count:", 0, 500)
    df = df.iloc[:max_data]
    df = df.drop(df.columns[1:13], axis=1)

    uploaded_kamusalay = st.file_uploader("Upload file CSV anda", key = 'kamusalay')
    if uploaded_kamusalay is not None:
        df_uploaded_kamusalay = pd.read_csv(uploaded_kamusalay, encoding = "ISO-8859-1", sep=',')

    df['text_only'] = df['Tweet'].apply(lambda x:remove_punct(x))  
    df.info()
    df['lowercase'] = df['text_only'].str.lower()
    df['tokenizing'] = df['lowercase'].apply(lambda x: tokenization(x.lower()))
    try: 
        df['normalization'] = df['tokenizing'].apply(lambda x: normalisasi(x))    
    except:
        df['normalization'] = df['tokenizing']
    df['stemming'] = df['normalization'].apply(lambda x: stemming(x))
    df['after_stopword'] = df['stemming'].apply(lambda x: remove_stopwords(x))
    df['detokenizing']=df['after_stopword'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))
    df['Cleaned_data'] = df['detokenizing'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ').str.replace('xf','')
    df
    if max_data > 0:
        st.markdown("""Here the result:""")    
        df[['Tweet','Cleaned_data']]
        df = df[['Tweet','Cleaned_data']]

#EXPORT TO CSV FILE
try:
    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')
    csv = convert_df(df)
    if not df.empty:
        st.download_button("Export result to CSV",csv,"cleaned_data.csv","text/csv",key='browser-data')

        #SAVE DATA TO SQLITE3
        if st.button("Save to database"):    
            conn = get_connection(URI_SQLITE_DB)
            init_db(conn)
            conn.execute('DELETE from result')
            for index,row in df.iterrows():
                conn.execute('INSERT INTO result values (?,?)', (row['Tweet'], row['Cleaned_data']))
                conn.commit()            
            display_data(conn)

            try:
                conn.close()
            except Exception:
                pass

except Exception:
    pass


















# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory #WORK
# factory = StopWordRemoverFactory() #WORK
# stop = factory.create_stop_word_remover() #WORK
# df['wo_stopwd'] = df['lower'].apply(lambda x: " ".join(stop.remove(x) for x in x.split())) #WORK