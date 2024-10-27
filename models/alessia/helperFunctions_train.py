import pandas as pd
import numpy as np

# Function for creating the dataset for training
def createDataSet_train(file):
    df = pd.read_csv(file)
    
    # Start time samples from zero
    df["5minute_intervals_timestamp"] = df["5minute_intervals_timestamp"] - df["5minute_intervals_timestamp"][0]

    # List to hold DataFrame indices
    carbIntake = []

    # Filtering for carbohydrate intake, bolus, and heart rate events
    filtered_df_carbs = df[df["carbInput"].notnull()]
    filtered_df_bolus = df[df["bolus"].notnull()]
    filtered_df_hr = df[df["hr"].notnull()]

    sampling_time = 5
    nbr_of_mins = 10
    nbr_of_hours = 3
    nbr_of_lags = 3
    nbr_of_time_steps = int(nbr_of_mins / sampling_time)
    nbr_of_total_samples = (60 // nbr_of_mins) * nbr_of_hours

    # Helper function to append related events within a range
    def append_if_in_range(subset_a, subset_b, distance=6, bolus=False):
        ranges = [(idx - distance, idx + distance) for idx in subset_a.index]

        def in_any_range(idx):
            return any(start <= idx <= end for start, end in ranges)

        valid_b = subset_b[subset_b.index.to_series().apply(in_any_range)]
        temp = pd.concat([subset_a, valid_b], join="outer").sort_index().drop_duplicates()

        return temp

    # Prepare sequences
    for index, row in filtered_df_carbs.iterrows():
        filtered_df_carbs = filtered_df_carbs.drop(filtered_df_carbs.index.to_list()[0])
        carbIntakePerIndex = []
        
        # Creating a sequence based on total samples
        for i in range(nbr_of_total_samples):
            try:
                carbIntakePerIndex.append(df.iloc[index + i * nbr_of_time_steps])
            except IndexError:
                break

        # Create DataFrame for current carbohydrate intake sequence
        carbIntakePerIndex_df = pd.DataFrame(carbIntakePerIndex)

        # Append related bolus and carb intake events within a given range
        carbIntakePerIndex_df = append_if_in_range(carbIntakePerIndex_df, filtered_df_carbs)
        carbIntakePerIndex_df = append_if_in_range(carbIntakePerIndex_df, filtered_df_bolus)

        # Creating lagged features for glucose (to track glucose trends)
        lagged_features = {f'glucose_lag_{lag}': df['cbg'].shift(lag * nbr_of_time_steps).bfill() for lag in range(1, nbr_of_lags)}
        carbIntakePerIndex_df = carbIntakePerIndex_df.assign(**lagged_features)

        # Drop any sequences with missing "cbg" values
        if carbIntakePerIndex_df["cbg"].isna().any():
            continue

        # Append the final prepared DataFrame for this sequence
        carbIntake.append(carbIntakePerIndex_df.fillna(0).drop(["5minute_intervals_timestamp"], axis=1))

    return carbIntake
