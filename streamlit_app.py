import streamlit as st
import pandas

st.title('Machinelearning App')
st.info('This is the machine learning app!')

df = pd.read_csc('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
df
