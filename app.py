

import numpy as np

import pandas as pd
import streamlit as st 
import json
import string
import pickle
import os

print("Directories are")
print( os.listdir())
# loading the saved model
loaded_model = pickle.load(open('/trained_model.sav', 'rb'))




def welcome():
    return "Welcome All"


def predict(place,place_df):
    with open('/svm_classifier_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    prediction=place+(clf.predict(place_df))[0]
    return prediction

  
def main():
    st.title("Named Entity Recognition")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Inhabitants ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    place = st.text_input("Place","Type Here")

    result=""
    if st.button("Predict"):
        place_df=preproessing(place)
        result=predict(place,place_df)
        print(result)
    
    st.success(result)

if __name__=='__main__':
    main()
    
    
    