#importing files
import streamlit as st
import numpy as np
import pandas as pd


#Info for the dashboard
st.title('Data Science Dashboard')
st.subheader('Dataset utilized: Iris dataset')

#loading the dataset
df = pd.DataFrame(pd.read_csv('iris_01.csv'))
st.dataframe(df)
st.caption('Raw Dataset')
