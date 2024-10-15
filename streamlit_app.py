import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('Machinelearning App')
st.info('This is the machine learning app!')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw
  
  st.write('**Y**')
  Y_raw = df.species
  Y_raw

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
    input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input Features'):
  st.write('**Input Penguin**')
  input_df
  st.write('**Combined Penguin Data**')
  input_penguins

# Encode x
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
input_raw = df_penguins[:1]
X = df_penguins[1:]

# Encode y
target_mapper = {
  "Adelie": 0,
  "Chinstrap": 1,
  "Gentoo": 2,
}

def target_encode(val):
  return target_mapper[val]

Y = Y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X**')
  input_raw
  st.write('**Encoded Y**')
  Y

# Model Training
clf = RandomForestClassifier()
clf = clf.fit(X, Y)

# Apply model to make prediction
prediction = clf.predict(input_raw)
prediction_proba = clf.predict_proba(input_raw)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.rename(columns={'0': 'Adelie', '1': 'Chinstrap', '2': 'Gentoo'})

st.subheader('Predicted Species')
penguin_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
df_prediction_proba
st.success(str(penguin_species[prediction][0]))


