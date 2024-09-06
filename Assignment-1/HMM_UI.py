import streamlit as st
import pandas as pd
import numpy as np
from HMM import HiddenMarkovModel, Viterbi, create_float_defaultdict

model = HiddenMarkovModel()
model.load()

st.set_page_config(
    page_title="POS Tagging with Hidden Markov Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('POS Tagger')
st.markdown('Interface to predict part of speech tag for each word of a given sentence using Hidden Markov Model')

image, input = st.columns(2)

with image:
    st.markdown(
        f'<img src="https://miro.medium.com/v2/resize:fit:600/1*qZELwIpKeEQ-j3EnRF-CrQ.jpeg">',
        unsafe_allow_html=True,
    )

with input:

    st.header("Predict Tags")
    sentence = st.text_input('Enter Input Sentence')

    if st.button("Predict POS Tags"):
        words = sentence.split()
        tags = model.predict(words)
        st.subheader("Result :")
        for i in range(len(words)):
            st.markdown(f"{words[i]}  ->  {tags[i]}")