import streamlit as st
import pandas as pd

st.title('Machinelearning App')
st.info('This is the machine learning app!')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df
