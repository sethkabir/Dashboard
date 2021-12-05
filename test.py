#THE TECHNOLOGIES USED TO CREATE THIS DASHBOARD ARE
#1. STREAMLIT - FOR THE USER INTERFACE
#2. PLOTLY - FOR BUILDING GRAPHS

#importing libraries
from typing import Optional
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


#PAGE TITLE AND MAIN HEADINGS FOR THE PROJECT
st.set_page_config(
    page_title='Dashboard',
)
st.title('Data Science Dashboard')
st.header('Dataset utilized: Iris dataset')



#IMPORTING DATASET
st.subheader('# Importing Dataset')
df = pd.DataFrame(pd.read_csv('IRIS.csv')) #Raw Dataset (creating a dataframe with the name df)
st.dataframe(df.head()) #display the first 5 rows
st.caption('Raw Dataset')



#DATA CLEANING
#Checking for Nan values
if(not df.isnull().values.any()):
    st.success('The dataset is clean, i.e. no Nan values or redundant features')
#checking for redundant features will be done manually, in this case this dataset did not have any reudundant features



#GENERAL INFORMATION ON THE DATASET
st.markdown('**_Some info on the the dataset_**')
st.write('1. Dataset total rows and columns: ', df.shape,) #describes the number of rows and columns
st.write('2. The description of the whole dataset: ', df.describe()) #describes general characteristic features for any dataset



#DATA VISUALIZATION 
st.subheader('# Data Visualization')
chart_data = pd.DataFrame(pd.read_csv('IRIS.csv'), columns=['sepal_length', 'sepal_width','petal_length','petal_width']) #creating a dataframe without the column "species"

#LINE GRAPH
st.write('**1. Line chart for the different features**')
st.line_chart(chart_data, height=300) #command to build a line graph 

#SCATTERPLOT GRAPH
st.write('**2. Scatterplot graphs for different features**')
cols = list(df.columns)
col1, col2 = st.columns(2) #this command is used 2 create 2 columns to insert the interactive buttons

with col1:
    x_val = st.selectbox('X-axis', options=cols)
with col2:
    y_val = st.selectbox('Y-axis', options=cols)
st.caption('Choose x and y axis for scatterplot graph')
plot = px.scatter(data_frame = df,x=x_val,y=y_val, color='species') #using plotly commands to build the scatterplot
st.plotly_chart(plot)

#HISTOGRAMS
st.write('**3. Histograms for different features**')
hist_cols = list(chart_data.columns)
hist_val = st.selectbox('X-axis', options=hist_cols)
hist_plot = px.histogram(data_frame=df,x=hist_val, color='species') #using plotly commands to build the histogram
st.plotly_chart(hist_plot)

#PIE CHART
st.write('**4. Pie chart for the number of samples identified as different labels**')
category_distribution = df.groupby(['species']).size().reset_index(name='counts')
fig = px.pie(category_distribution, values='counts', names='species') #using plotly commands to build the pie chart
st.plotly_chart(fig)



#DATA CLASSIFICATION
st.subheader('# Data Classification')

#LOGISTIC REGRESSION 
st.write('**1. Logistic Regression**')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Seperating the data into dependent and independent variables
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
st.write('Classification Report')
st.text(classification_report(y_test, y_pred))
st.write('Confusion Matrix')
st.table(confusion_matrix(y_test, y_pred))
st.write('Heatmap for confusion matrix')
fig = px.imshow(confusion_matrix(y_test, y_pred))
st.plotly_chart(fig)

# Accuracy score
from sklearn.metrics import accuracy_score
st.write('Accuracy ', accuracy_score(y_pred,y_test))