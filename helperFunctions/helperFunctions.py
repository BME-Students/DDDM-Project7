import pandas as pd
import numpy as np
from IPython.display import display

## Create Data Set
def createDataSet(file):
    df = pd.read_csv(file)
    # Start time samples from zero
    df["5minute_intervals_timestamp"] = df["5minute_intervals_timestamp"] - df["5minute_intervals_timestamp"][0]
    # List of DataFrame indexes 
    carbIntake = []

    # Filter for carb intake, where carb intake is not null
    filtered_df_carbs = df[df["carbInput"].notnull()]
    filtered_df_bolus= df[df["bolus"].notnull()]
    filtered_df_hr = df[df["hr"].notnull()]
    sampling_time = 5
    nbr_of_mins = 10
    nbr_of_hours = 3
    nbr_of_lags = 3
    nbr_of_time_steps =int(nbr_of_mins/sampling_time)
    nbr_of_samples_per_h = int(np.floor(60/nbr_of_mins))
    nbr_of_total_samples = nbr_of_samples_per_h*nbr_of_hours

    def append_if_in_range(subset_a, subset_b, distance=6, bolus = False):
        ranges = [(idx - distance, idx + distance) for idx in subset_a.index]  # Create ranges around each index in A
    
        # Function to check if the index is within any range
        def in_any_range(idx):
            return any(start <= idx <= end for start, end in ranges)
        
        # Create a new dataframe for valid entries in subset B
        valid_b = subset_b[subset_b.index.to_series().apply(in_any_range)]
        
        # Append the valid rows from subset B to subset A
        temp = pd.concat([subset_a, valid_b], join="outer").sort_index().drop_duplicates()

        # List of index values where we want to split the DataFrame
        split_indices = valid_b.index.to_list() # The DataFrame will be split at index 3 and 6
        # Initialize the starting point of the split
        start_idx = subset_a.index.to_list()[0]
        # Initialize an empty list to hold the sub-DataFrames
        sub_dfs = []

        # Iterate through the split indices and slice the DataFrame
        for idx in split_indices:
            sub_dfs.append(temp.loc[start_idx:idx-1].copy())
            start_idx = idx

        # Append the last section of the DataFrame (from last split point to end)
        sub_dfs.append(temp.loc[start_idx:].copy())

        # Show the resulting sub-DataFrames
        if not bolus:
            for sub_df in sub_dfs:
                values = sub_df.index.to_list()
                sub_df["minAfterMeal"] = [(x-values[0])*5 for x in values]
        else:
            for sub_df in sub_dfs:
                values = sub_df.index.to_list()
                sub_df["minAfterBolus"] = [(x-values[0])*5 for x in values]

        result = pd.concat(sub_dfs)
        
        return result

    for index, row in filtered_df_carbs.iterrows():
        filtered_df_carbs = filtered_df_carbs.drop(filtered_df_carbs.index.to_list()[0])
        carbIntakePerIndex = []
        for i in range(nbr_of_total_samples):
            try:
                carbIntakePerIndex.append(df.iloc[index+i*nbr_of_time_steps])
            except:
                print(f"Sample {index+i*nbr_of_samples_per_h} is out of range")
                break
        # Create the Data Frame
        carbIntakePerIndex_df = pd.DataFrame(carbIntakePerIndex)
            
        # Include all indeces of further carb intakes between the indeces
        carbIntakePerIndex_df = append_if_in_range(carbIntakePerIndex_df, filtered_df_carbs)
        # carbIntakePerIndex_df = append_if_in_range(carbIntakePerIndex_df, filtered_df_bolus)
        

        # Creating lagged features for glucose which allows model to consider how glucose concentration has been trending
        lagged_features = {f'glucose_lag_{lag}': df['cbg'].shift(lag*nbr_of_time_steps).bfill() for lag in range(1, nbr_of_lags)}
        carbIntakePerIndex_df = carbIntakePerIndex_df.assign(**lagged_features)
        if carbIntakePerIndex_df["missing_cbg"].any() > 0:
            continue

        carbIntake.append(carbIntakePerIndex_df.fillna(0).drop(["5minute_intervals_timestamp"], axis=1).set_index((x-carbIntakePerIndex_df.index.tolist()[0])*5 for x in carbIntakePerIndex_df.index.tolist()))
    return carbIntake
    

if __name__ == "__main__":
    # Load the dataset
    file = "data/Ohio2020_processed/train/540-ws-training_processed.csv"
    
    data = createDataSet(file)
    display(data[0])