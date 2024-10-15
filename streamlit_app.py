import streamlit as st
import pandas as pd

st.title('Machinelearning App')
st.info('This is the machine learning app!')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X = df.drop('species', axis=1)
  X
  
  st.write('**Y**')
  Y = df.species
  Y

with st.expander('Data Visualizations'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

with st.sidebar:
    st.header('Input Fearures')
    island = st.selectbox('Island', ('Torgersen', 'Biscoe', 'Dream'))
    gender = st.selectbox('Gender', ('male', 'female'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
