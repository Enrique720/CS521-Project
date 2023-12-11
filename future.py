import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam 

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob 


def futureForecast(df, col, n_input, n_features, forecast_timeperiod, model):

    x_input = np.array(df[len(df)-n_input:][col])

    temp_input=list(x_input)

    lst_output=[]
    i=0

    while(i < forecast_timeperiod):

        if(len(temp_input) > n_input):

            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape((1, n_input, n_features))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            temp_input = temp_input[1:]
            lst_output.append(yhat[0][0])

            i=i+1

        else:
            x_input = x_input.reshape((1, n_input, n_features))
            yhat = model.predict(x_input, verbose=0)
            #print(yhat[0])
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])

            i=i+1
            
    return lst_output

def Sequential_Input_LSTM(df, input_sequence):
    df_np = df.to_numpy()
    X = []
    y = []
    
    for i in range(len(df_np) - input_sequence):
        row = [a for a in df_np[i:i + input_sequence]]
        X.append(row)
        label = df_np[i + input_sequence]
        y.append(label)
        
    return np.array(X), np.array(y)

def get_groups(dateList, values):
    groups = []
    groups_datetime = []
    gap_lengths = []  # List to store the length of gaps
    index_stamp = 0

    for i in range(1, len(dateList)):
        time_difference = (dateList[i] - dateList[i-1]).total_seconds() / 60

        if time_difference > 15:
            temp_df = values[index_stamp:i]
            date_df = dateList[index_stamp:i]

            index_stamp = i
            groups.append(temp_df)
            groups_datetime.append(date_df)

            # Calculate the length of the gap and store it
            gap_length = time_difference - 15  # Subtract the 15 minute threshold
            gap_lengths.append(gap_length/15)

        if i == len(dateList) - 1:
            temp_df = values[index_stamp:i+1]
            date_df = dateList[index_stamp:i+1]

            groups.append(temp_df)
            groups_datetime.append(date_df)

    return groups, groups_datetime, gap_lengths




station = "montezuma"
parameter = "DO2"


stations = [ 'montezuma', 'sapello',  'lourdes' , 'firest']
parameters = ['DO2', 'fdom', 'turbidity', 'ph', 'spcond', 'temperature']



for station in stations:
    for parameter in parameters:

        # Load the dataset
        df = pd.read_csv(f'data/{parameter}_{station}.csv')

        df['DateTime'] = pd.to_datetime(df['DateTime'])

        groups, groups_datetime, gap_lengths = get_groups(df['DateTime'], df[parameter])

        # Find the largest group
        largest_group_index = max(range(len(groups)), key=lambda i: len(groups[i]))
        largest_group = groups[largest_group_index]
        largest_group_dates = groups_datetime[largest_group_index]
        largest_group_gap_length = gap_lengths[largest_group_index] if largest_group_index < len(gap_lengths) else None

        print(f"Largest group shape: {largest_group.shape}")
        print(f"Largest group gap length: {largest_group_gap_length}")

        if largest_group_gap_length == None:
            largest_group_gap_length = 1

        # Create a DataFrame from the largest group
        df_largest_group = pd.DataFrame({
            'DateTime': largest_group_dates,
            'Value': largest_group
        })

        # Interpolating the data

        # Set the 'DateTime' column as the index
        df_largest_group.set_index('DateTime', inplace=True)

        # Determine the new date range, with 5-minute intervals
        start = df_largest_group.index.min()
        end = df_largest_group.index.max()
        new_date_range = pd.date_range(start=start, end=end, freq='5T')

        # Reindex the DataFrame to the new date range and interpolate the missing values
        df_augmented = df_largest_group.reindex(new_date_range)
        df_augmented['Value'] = df_augmented['Value'].interpolate()

        # Reset index to make 'DateTime' a column again
        df_augmented.reset_index(inplace=True)
        df_augmented.rename(columns={'index': 'DateTime'}, inplace=True)

        print(df_augmented.head)

        n_input = 20  # try also 144, 72, 24
        # Data Preparation
        X, y = Sequential_Input_LSTM(df_augmented["Value"], n_input)
        n_features =1 


        # Split Data for Training and Validation
        train_size = int(len(X) * 0.8)  # 80% for training
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]

        # Model Definition
        model = Sequential()
        model.add(InputLayer((n_input, n_features)))
        model.add(LSTM(100, return_sequences=True))     
        model.add(LSTM(100, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), 
                    metrics=[RootMeanSquaredError()])

        # Training
        early_stop = EarlyStopping(monitor='val_loss', patience=2)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[early_stop])

        # Forecasting
        forecast_timeperiod = largest_group_gap_length*3  # e.g., forecast next 240 time steps
        n_input = 20
        n_features = 1

        forecast_output = futureForecast(df_augmented, 
                                        'Value', 
                                        n_input, 
                                        n_features, 
                                        forecast_timeperiod, 
                                        model)


        last_10_days = df_augmented['Value'][len(df_augmented) - 240:].tolist()

        next_10_days = pd.DataFrame(forecast_output, columns = ['FutureForecast'])

        plt.figure(figsize = (15,5))

        hist_axis = len(last_10_days)
        forecast_axis = hist_axis + len(next_10_days)

        plt.plot(np.arange(0,hist_axis),last_10_days, color = 'blue')
        plt.plot(np.arange(hist_axis,forecast_axis),next_10_days['FutureForecast'].tolist(), color = 'orange')

        plt.title(f'LSTM Forecast for Next {largest_group_gap_length*3} intervals')
        plt.xlabel('Hours')
        plt.ylabel(f'{parameter}')

        # save the figure
        plt.savefig(f'plots/forecast/{station}/{parameter}.png')
        plt.savefig(f'plots/forecast/{station}/{parameter}.pdf')