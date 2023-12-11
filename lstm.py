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

stations = ["firest"]
#parameter = "DO2"

#stations = [ 'montezuma', 'sapello',  'lourdes' , 'firest']
parameters = ['DO2', 'fdom', 'turbidity', 'ph', 'spcond', 'temperature']
#parameters = ['ph']

for station in stations:
    for parameter in parameters:

        # Load the dataset
        df = pd.read_csv(f'data2/{parameter}_{station}.csv')

        df['DateTime'] = pd.to_datetime(df['DateTime'])

        groups, groups_datetime, gap_lengths = get_groups(df['DateTime'], df[parameter])

        # Find the largest group
        largest_group_index = max(range(len(groups)), key=lambda i: len(groups[i]))
        largest_group = groups[largest_group_index]
        largest_group_dates = groups_datetime[largest_group_index]
        largest_group_gap_length = gap_lengths[largest_group_index] if largest_group_index < len(gap_lengths) else None

        print(f"Largest group shape: {largest_group.shape}")
        print(f"Largest group gap length: {largest_group_gap_length}")

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

        X, y = Sequential_Input_LSTM(df_augmented["Value"], n_input)

        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        total_data = X.shape[0]  # Total number of data points
        train_idx = int(total_data * 0.7)  # 70% for training
        val_idx = train_idx + int(total_data * 0.2)  # Additional 20% for validation

        # Split the data
        X_train, y_train = X[:train_idx], y[:train_idx]  # 70% for training
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]  # Next 20% for validation
        X_test, y_test = X[val_idx:], y[val_idx:]  # Remaining 10% for testing

        # Print the shapes of the new splits
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        ## Creating the model

        n_features = 1                        

        model1 = Sequential()

        model1.add(InputLayer((n_input,n_features)))
        model1.add(LSTM(100, return_sequences = True))     
        model1.add(LSTM(100, return_sequences = True))
        model1.add(LSTM(50))
        model1.add(Dense(100, activation = 'relu'))
        model1.add(Dense(1, activation = 'linear'))

        model1.summary()

        early_stop = EarlyStopping(monitor = 'val_loss', patience = 2)

        model1.compile(loss = MeanSquaredError(), 
                    optimizer = Adam(learning_rate = 0.0001), 
                    metrics = RootMeanSquaredError())


        # Fit and save model if it doesn't exist
        print("Model not Found - Training Model")
        model1.fit(X_train, y_train, 
                validation_data = (X_val, y_val), 
                epochs = 50, 
                callbacks = [early_stop])
        save_model(model1, f"LSTM_Models/{station}/{parameter}_{n_input}.keras")
        print("model1 history: ")
        print(model1.history.history)
        losses_df1 = pd.DataFrame(model1.history.history)
        losses_df1.plot(figsize = (10,6))
        plt.title(f"Model History for {parameter} in {station}")
        plt.xlabel("Epochs")
        plt.ylabel("Error/Loss")
        plt.savefig(f"plots/loss/{station}/{parameter}{n_input}.png")


        test_predictions1 = model1.predict(X_test).flatten()

        X_test_list = []
        for i in range(len(X_test)):
            X_test_list.append(X_test[i][0])
            
        test_predictions_df1 = pd.DataFrame({'X_test':list(X_test_list), 
                                            'LSTM Prediction':list(test_predictions1)})

        print(test_predictions_df1.head())

        # Test model performance

        test_predictions_df1.plot(figsize = (15,6))
        plt.title(f"Prediction vs Real - {station}")
        plt.xlabel("Time")
        plt.ylabel(f"{parameter}")
        plt.savefig(f"plots/prediction/{station}/{parameter}_{n_input}.png")