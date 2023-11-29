

import numpy as np

import pandas as pd
import streamlit as st 
import json
import string
import pickle
import os

import nltk
from nltk.tokenize import word_tokenize




nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

All_tags={'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
 'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
 'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
 'WP': 44, 'WP$': 45, 'WRB': 46}




def welcome():
    return "Welcome All"


def predict(sentence):
    # Example sentence
    # sentence = "Educative Answers is a free web encyclopedia written by devs for devs."

    # Tokenization
    tokens = word_tokenize(sentence)

    # POS tagging
    pos_tags = nltk.pos_tag(tokens)

    digits=[]
    case=[]
    title=[]
    pos=[]
    for i in pos_tags:
        pos.append(All_tags[i[1]])
        if i[0].isdigit()==True:
                digits.append(1)
                case.append(0)
                title.append(0)
        else:
            digits.append(0)
            if i[0][0].isupper():
                case.append(1)
            else:
                case.append(0)
            if i[0].istitle():
                title.append(1)
            else:
                title.append(0)
    X_test=pd.DataFrame(columns=['POS','Case','Digits','Title'])

    X_test['POS']=pos
    X_test['Case']=case
    X_test['Digits']=digits
    X_test['Title']=title
    prediction=loaded_model.predict(X_test)
    return prediction

  
def main():
    st.title("Named Entity Recognition")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Inhabitants ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    sentence = st.text_input("Sentence","Type Here")

    result=""
    if st.button("Predict"):
        result=predict(sentence)
        print(result)
    
    st.success(result)

if __name__=='__main__':
    main()
    
    
    