import numpy as np
import streamlit as st
from io import BytesIO
import wave
import os
import matplotlib.pyplot as plt
import requests
import subprocess
import time

st.set_page_config(
    page_title="Predict Text transcriptions using SpeechTokenizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Speech Tokenizer')
st.markdown('Interface to predict text transcriptions of speech input using SpeechTokenizer')

image, input = st.columns(2)

with image:
    st.markdown(
        f'<img src="https://nordicapis.com/wp-content/uploads/5-Best-Speech-to-Text-APIs-2-e1615383933700-1024x573.png">',
        unsafe_allow_html=True,
    )

with input:

    st.header("Transcribe Audio")
    audio = st.file_uploader("Upload an audio file", type=["flac"])

    if st.button("Transcribe Audio"):
        f = open('/raid/speech/aryan/input.flac', 'wb')
        f.write(audio.getvalue())
        f.close()
        env = os.environ.copy()
        #subprocess.run("python3 inference_st.py hparams/inference_st.yaml".split())
        result = subprocess.check_output(["python3", "inference_st.py", "hparams/inference_st.yaml"],text=True,env=env)
        #st.write(result)
        # Debugging information
        #st.text("Subprocess stdout:")
        #st.text(result.stdout)
        #st.text("Subprocess stderr:")
        #st.text(result.stderr)
        #time.sleep(10)
        fin = open('output.txt', 'r')
        st.subheader("Result :")#, fin.read()) # streamlit run app.py --server.fileWatcherType none
        st.write(fin.read())
