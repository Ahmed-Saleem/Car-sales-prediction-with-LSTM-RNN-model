import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten
import matplotlib.pyplot as plt

dataset=pd.read_csv('norway_new_car_sales_by_month.csv')
dataset['Date']=pd.to_datetime(dataset['Date'],format='%b-%y')
n_intervals=3
def prepare_data(timeseries_data, n_intervals):
    X,y=[],[]
    length= len(timeseries_data)
    while 1:
        if length % n_intervals >0:
            length= length-2
        else:
            break
    for i in range(length):
        i_end=i+n_intervals
        if i_end> len(timeseries_data):
            break
        X.append(timeseries_data.iloc[i:i_end])
        y.append(timeseries_data.iloc[i_end])
    return np.array(X), np.array(y)
X,y= prepare_data(dataset['Quantity'],n_intervals)

#reshape into 3D
n_features=1
X=X.reshape((X.shape[0], X.shape[1], n_features))

#define model
model= Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_intervals, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=300, verbose=0)

# demonstrate prediction for next 10 months
n_months=10
x_input = np.append(X[-1,1:],y[-1])
temp_input=list(x_input)
lst_output=[]
i=0
while(i<n_months):
    
    if(len(temp_input)>3):
        x_input=np.array(temp_input[1:])
        print("{} month input {}".format(i,x_input))
        #print(x_input)
        x_input = x_input.reshape((1, n_intervals, n_features))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} month output {}".format(i,yhat))
        temp_input.append(yhat[0][0])
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.append(yhat[0][0])
        i=i+1
    else:
        x_input = x_input.reshape((1, n_intervals, n_features))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        i=i+1
print(lst_output)

dateList=pd.date_range(start=dataset.Date.iloc[-1], periods=n_months, freq='M')

plt.plot(dataset['Date'],dataset['Quantity'])
plt.plot(dateList,lst_output)
plt.show()
    
