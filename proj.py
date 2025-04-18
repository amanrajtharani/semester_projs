import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from tensorflow.keras.models import load_model
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import streamlit as st

st.title("Stock predictor")
start = '2010-01-01'
end = '2022-12-31'


user_input = st.text_input("Enter stock ticker","AAPL")
df = yf.download(user_input, start, end)



st.subheader("Data")
st.write(df.describe())

#visualization
st.subheader("Closing price vs time chart")
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
st.pyplot(fig)


st.subheader("closing price vs time chart with ma100 and ma200")
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig1 = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
st.pyplot(fig1)

#split into train and test
train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


scaler = MinMaxScaler(feature_range=(0,1))
training_array = scaler.fit_transform(train)  

x_train = []
y_train = []

for i in range(100, training_array.shape[0]):
    x_train.append(training_array[i-100:i])
    y_train.append(training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)



model = load_model('keras_model.h5')


past100 = train.tail(100)
final_df = pd.concat([past100, test], ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0]) 


x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)


scaler.scale_

scale_factor = 1/0.00689823
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


st.subheader("prediction vs original")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)