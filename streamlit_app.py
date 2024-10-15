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
    sex = st.selectbox('Gender', ('male', 'female'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172, 231, 201)
    body_mass_g = st.slider('Body mass (gm)', 2700, 6300, 4207)

    data = {
      'island': island,
      'sex': sex,
      'bill_length_mm': bill_length_mm,
      'bill_depth_mm': bill_depth_mm,
      'flipper_length_mm': flipper_length_mm,
      'body_mass_g': body_mass_g 
    }

    input_df = pd.DataFrame(data, index=[0])
    input_penguins = pd.concat([input_df, X], axis=0)

with st.expander('Input Features'):
  st.write('**Input Penguin**')
  input_df
  st.write('**Combined Penguin Data**')
  input_penguins

