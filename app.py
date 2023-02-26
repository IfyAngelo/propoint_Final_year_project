import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizer.from_pretrained('DataAngelo/propoint_Final_project')
    model = BertForSequenceClassification.from_pretrained('DataAngelo/propoint_Final_project')
    return tokenizer, model

tokenizer, model = get_model()

user_input = st.text_area('Enter Choice: ')
button = st.button('ProPoint')

d = {
    0: 'Poor',
    1: 'Bad',
    2: 'Average',
    3: 'Good',
    4: 'Very Good',
    5: 'Excellent'
}

if user_input and button:
    text = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')
    output = model(**text)

    st.write('Logits: ', output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(), axis=1)
    
    st.write('ProPoint Predicts: ', d[y_pred[0]])