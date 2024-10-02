import streamlit as st
from CRF import FeatureExtractor
import nltk
import pycrfsuite
from nltk.corpus import brown
import numpy as np

nltk.download('brown')
nltk.download('universal_tagset')
dataset = list(brown.tagged_sents(tagset='universal'))
train_vocab = {word.lower() for sent in dataset for word, _ in sent}
feature_extractor = FeatureExtractor(train_vocab)
tagger = pycrfsuite.Tagger()
tagger.open('crf_pos_tagger_cv.model')

st.set_page_config(
    page_title="POS Tagging with CRF Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('POS Tagger')
st.markdown('Interface to predict part of speech tag for each word of a given sentence using CRF Model')

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
        X = feature_extractor.sent2features(words)
        y_pred = [tagger.tag(feature_extractor.sent2features(sent)) for sent in [words]]
        st.subheader("Result :")
        for i in range(len(words)):
            st.markdown(f"{words[i]}  ->  {y_pred[0][i]}")