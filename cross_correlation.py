import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from datetime import datetime


import pandas as pd

start_datetimes = {
    "fdom": datetime(2022, 7, 12),
    "ph": datetime(2022, 7, 12),
    "temperature": datetime(2022, 5, 15),
    "DO2": datetime(2022, 5, 15),
    "turbidity": datetime(2022, 7, 15),
    "spcond": datetime(2022, 5, 15)
}

end_datetimes = {
    "fdom": datetime(2022, 8, 8),
    "ph": datetime(2022, 8, 8),
    "temperature": datetime(2022, 6, 15),
    "DO2": datetime(2022, 6, 19),
    "turbidity": datetime(2022, 8, 8),
    "spcond": datetime(2022, 6, 19)
}


stations = ["firest", "montezuma", "lourdes"]
parameters = ["turbidity", "fdom", "temperature", "DO2", "turbidity", "spcond"]


for parameter in parameters:
    
    pd_list = []
    
    for station in stations: 
        file_path = f"data/{parameter}_{station}.csv"
        pd_list.append(pd.read_csv(file_path))

    start_date = start_datetimes[parameter]
    end_date = end_datetimes[parameter]

    print(f"start date : {start_date}")
    print(f"end date : {end_date}")    

    time_lag = 4  # Starting with 1 station
    it = 0

    for i, df in enumerate(pd_list):
        # Finding the NaN values in the DataFrame
        nan_mask = df.isna()
        if nan_mask.any().any():  # Check if there is any NaN value in the DataFrame
            print(f"NaN values found in DataFrame {i}")
            nan_indices = nan_mask[nan_mask].index.tolist()
            print(f"Indices with NaN values: {nan_indices}")
        else:
            print(f"No NaN values found in DataFrame {i}")

    for i in range(len(pd_list)):
        pd_list[i]['DateTime'] = pd.to_datetime(pd_list[i]['DateTime'])
        pd_list[i].set_index('DateTime', inplace=True)
        pd_list[i] = pd_list[i].loc[start_date:end_date]
        complete_index = pd.date_range(start=start_date, end=end_date, freq='15T')
        pd_list[i] = pd_list[i].reindex(complete_index)
        pd_list[i] = pd_list[i].interpolate(method='linear')

    for i in range(len(pd_list)):
        fin = len(pd_list[i]) - (len(pd_list)-i)*time_lag
        pd_list[i] = pd_list[i][it*time_lag:fin]
        it+= 1

    print("----------")
    for list in pd_list:
        print(len(list))
    print("----------")


    # Verifying lengths
    verif = len(pd_list[0])
    for i in range(1,len(pd_list)):
        if (verif != len(pd_list[i])):
            exit(1)

    stacked_arrays = np.vstack([list[parameter].to_numpy() for list in pd_list])

    print("Starting cross correlation")
    cross_correlation_matrix = np.corrcoef(stacked_arrays)
    print("Finished Cross Correlation")


    print(cross_correlation_matrix)
    print(cross_correlation_matrix.shape)
 
    plt.figure()

    plt.imshow(cross_correlation_matrix, cmap='hot', interpolation='nearest')
    # Adding a color bar to understand the values
    plt.colorbar()

    # Optionally, add titles or labels here
    plt.title("Cross-Correlation Matrix")
    tick_positions = range(len(stations))  # Assuming one tick per station
    plt.xticks(tick_positions, stations)  # Set x-axis labels
    plt.yticks(tick_positions, stations)  # Set y-axis labels


    # Save the plot as an image file
    plt.savefig(f"plots/cross/{parameter}_{time_lag*15}.png")
    plt.close()