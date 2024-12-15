import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display

def time_of_day_one_hot(df, timestamp_column):
    """
    This function takes a DataFrame with a column containing timestamps and creates
    one-hot encoded columns for different times of day: Morning, Lunch, Afternoon, Evening, Night.

    :param df: Input DataFrame
    :param timestamp_column: The name of the timestamp column in the DataFrame
    :return: DataFrame with one-hot encoded time of day columns
    """
    
    # Specify the correct datetime format if you know the format of the timestamp
    # For example, if the timestamp format is "yyyy-mm-dd hh:mm:ss", we specify that
    datetime_format = "%Y-%m-%d %H:%M:%S"  # Change this if your format differs
    
    # Convert the timestamp column to datetime using the specified format
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], format=datetime_format, errors='coerce')  # Coerce errors if any

    # Define the time periods
    def get_time_of_day(timestamp):
        if timestamp.hour >= 6 and timestamp.hour < 12:
            return 'morning'
        elif timestamp.hour >= 12 and timestamp.hour < 14:
            return 'lunch'
        elif timestamp.hour >= 14 and timestamp.hour < 18:
            return 'afternoon'
        elif timestamp.hour >= 18 and timestamp.hour < 21:
            return 'evening'
        else:
            return 'night'

    # Apply the function to determine time of day and create the 'time_of_day' column
    df['time_of_day'] = df[timestamp_column].apply(get_time_of_day)

    # One-hot encode the 'time_of_day' column
    df_one_hot = pd.get_dummies(df['time_of_day'], prefix='ToD')

    # Concatenate the one-hot encoded columns to the original DataFrame
    df = pd.concat([df, df_one_hot], axis=1)

    # Optionally, drop the 'time_of_day' column if it's no longer needed
    df = df.drop('time_of_day', axis=1)

    return df

def add_minutes_after_finger(df):
    """
    Adds a column to the DataFrame with the `finger` prick value for 2 hours after each `finger` prick reading,
    ensuring that new `finger` values within this period overwrite previous values.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the original data, including 'finger' column.
    
    Returns:
    pd.DataFrame: Updated DataFrame with new 'finger_extended' column.
    """
    # Initialize the new column with NaNs
    df['finger_extended'] = np.nan
    
    # Define the number of rows to carry forward the 'finger' value (2 hours = 24 rows)
    rows_to_extend = 24

    # Variable to keep track of the current `finger` value to propagate
    last_finger_value = np.nan
    remaining_rows = 0  # Tracks how many rows to continue applying the last `finger` value

    # Iterate through each row to set `finger_extended` values
    for i in range(len(df)):
        if not pd.isnull(df.loc[i, 'finger']):  # A new `finger` value is detected
            # Update the current `finger` value and reset remaining rows to 24
            last_finger_value = df.loc[i, 'finger']
            remaining_rows = rows_to_extend

        # Set the `finger_extended` value if within the extension period
        if remaining_rows > 0:
            df.loc[i, 'finger_extended'] = last_finger_value
            remaining_rows -= 1  # Decrement the rows remaining for this `finger` value

    return df


def add_minutes_after_meal_and_meal_size(df):
    """
    Adds two columns: 'minutesAfterMeal' and 'mealSize'. 
    - 'minAfterMeal' tracks the minutes passed since the last 'carbInput', reset every 6 hours or when a new 'carbInput' is encountered.
    - 'mealSize' repeats the 'carbInput' until 3 hours (180 minutes) is reached.
    :param df: DataFrame containing the dataset with a 'carbInput' column
    :return: DataFrame with the new 'minutesAfterMeal' and 'mealSize' columns
    """
    # Initialize the 'minutesAfterMeal' and 'mealSize' columns
    df['minAfterMeal'] = 0
    df['mealSize'] = None  # Will hold the carbInput until the counter resets

    # Variable to track the minutes after the last carbInput and mealSize
    minutes_counter = 0
    last_meal_size = None
    
    # Iterate through each row
    for i in range(len(df)):
        if pd.notna(df.loc[i, 'carbInput']):  # If there is a carbInput
            minutes_counter = 0  # Reset the counter when a new carbInput appears
            last_meal_size = df.loc[i, 'carbInput']  # Set the mealSize to the current carbInput
        else:
            minutes_counter += 5  # Otherwise, increase the counter by 5 minutes (since each sample is 5 minutes apart)

        # If minutes exceed 3 hours (180 minutes), reset to 0
        if minutes_counter > 180:
            minutes_counter = 0  # Reset the counter when exceeding 6 hours
            last_meal_size = None  # Reset mealSize until the next carbInput
        
        # Assign the minutes count and mealSize to the respective columns
        df.loc[i, 'minutesAfterMeal'] = minutes_counter
        df.loc[i, 'mealSize'] = last_meal_size

    return df

def add_minutes_after_bolus_and_bolus_size(df):
    """
    Adds two columns: 'minutesAfterBolus' and 'bolusSize'. 
    - 'minAfterBolus' tracks the minutes passed since the last 'bolus', reset every 4 hours or when a new 'bolus' is encountered.
    - 'bolusSize' repeats the 'bolus' until 3 hours (180 minutes) is reached.
    :param df: DataFrame containing the dataset with a 'bolus' column
    :return: DataFrame with the new 'minutesAfterBolus' and 'bolusSize' columns
    """
    # Initialize the 'minutesAfterMeal' and 'mealSize' columns
    df['minAfterBolus'] = 0
    df['bolusSize'] = None  # Will hold the carbInput until the counter resets


    # Variable to track the minutes after the last carbInput and mealSize
    minutes_counter = 0
    last_bolus_size = None
    
    # Iterate through each row
    for i in range(len(df)):
        if pd.notna(df.loc[i, 'bolus']):  # If there is a carbInput
            minutes_counter = 0  # Reset the counter when a new carbInput appears
            last_bolus_size = df.loc[i, 'bolus']  # Set the mealSize to the current carbInput
        else:
            minutes_counter += 5  # Otherwise, increase the counter by 5 minutes (since each sample is 5 minutes apart)

        # If minutes exceed 3 hours (180 minutes), reset to 0
        if minutes_counter > 180:
            minutes_counter = 0  # Reset the counter when exceeding 6 hours
            last_bolus_size = None  # Reset mealSize until the next carbInput
        
        # Assign the minutes count and mealSize to the respective columns
        df.loc[i, 'minutesAfterBolus'] = minutes_counter
        df.loc[i, 'bolusSize'] = last_bolus_size
    
    return df

# Updated data preparation function to include all relevant columns
def prepare_features_target(data_list, feature_columns):
    X = []
    y = []
    
    for df in data_list:
        # Select the specified columns and flatten them into a single feature vector
        feature_vector = df[feature_columns].values.flatten()  # Shape: (24 * len(feature_columns),)
        
        # Append the feature vector to X
        X.append(feature_vector)
        
        # Define target as the last value in 'cbg' (or another target)
        target_value = df['cbg'].values[-1]
        y.append(target_value)
    
    # Convert to numpy arrays for compatibility with sklearn
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def createDataSet(file, scaler = None):
    # Load the dataset
    df = pd.read_csv(file)
    df = add_minutes_after_finger(df)
    df = add_minutes_after_meal_and_meal_size(df)
    df = add_minutes_after_bolus_and_bolus_size(df)
    # Blood Glucose Change Rate: Calculating the rate of change in blood glucose levels over various intervals 
    # can be an important feature for predicting future trends.
    df['cgm_rate_5min'] = df['cbg'].diff(periods=1) / 5  # Change in CGM per minute
    df['cgm_rate_15min'] = df['cbg'].diff(periods=3) / 15
    df = df.drop(["5minute_intervals_timestamp", "hr", "basal", "gsr"], axis=1)
    
    # Scale the whole dataset
    # Data is scaled using MinMaxScaler to ensure that all features are within the same range, typically between 0 and 1.
    # if not scaler:
    #     scaler = MinMaxScaler().set_output(transform="pandas")
    #     df[df.columns] = scaler.fit_transform(df[df.columns])
    # else:
    #     df[df.columns] = scaler.transform(df[df.columns])
    
    
    # List of DataFrame indexes 
    carbIntake = []

    # Filter for carb intake, where carb intake is not null
    filtered_df_carbs = df[df["carbInput"].notnull()]
    filtered_df_bolus = df[df["bolus"].notnull()]

    # Variables used for selecting the correct timestamps.
    sampling_time = 5
    nbr_of_mins = 5
    nbr_of_hours = 4
    nbr_of_lags = 3
    nbr_of_time_steps = int(nbr_of_mins/sampling_time)
    nbr_of_samples_per_h = int(np.floor(60/nbr_of_mins))
    nbr_of_total_samples = int(nbr_of_samples_per_h*nbr_of_hours)

    # Check for events (finger, bolus, carbInput) within 30 minutes of each other
    for i, row in filtered_df_carbs.iterrows():
        if pd.notnull(row["bolus"]) or pd.notnull(row["finger"]) or pd.notnull(row["carbInput"]):
            # Define a 10-minute window before and after the current index (8 rows in either direction)
            start_window = max(i - 2, 0)
            end_window = min(i + 2, len(df) - 1)

            # Check if "finger", "bolus", and "carbInput" are present within this window
            window_df = df.loc[start_window:end_window]
            if (
                window_df["bolus"].notnull().any() and
                window_df["finger"].notnull().any() and
                window_df["carbInput"].notnull().any()
            ):
                # Find the earliest occurrence of any of the three events
                first_event_index = min(
                    window_df[window_df["bolus"].notnull()].index[0],
                    window_df[window_df["finger"].notnull()].index[0],
                    window_df[window_df["carbInput"].notnull()].index[0]
                )

                # Set the start index at the first event
                start_index = first_event_index
        # start_index = i
            # Collect data starting from the identified start_index
                carbIntakePerIndex = []
                for j in range(nbr_of_total_samples + 8):
                    try:
                        carbIntakePerIndex.append(df.iloc[start_index + j * nbr_of_time_steps])
                    except IndexError:
                        print(f"Sample {start_index + j * nbr_of_samples_per_h} is out of range")
                        break

                # Create DataFrame from collected samples
                carbIntakePerIndex_df = pd.DataFrame(carbIntakePerIndex)

                # Add lagged glucose features
                lagged_features = {f'glucose_lag_{lag}': df['cbg'].shift(lag * nbr_of_time_steps).bfill() for lag in range(1, nbr_of_lags)}
                carbIntakePerIndex_df = carbIntakePerIndex_df.assign(**lagged_features)
                
                # Skip if there are missing values in 'missing_cbg' column
                if carbIntakePerIndex_df["missing_cbg"].any() > 0:
                    continue

                # Finalize DataFrame adjustments
                carbIntake.append(
                    carbIntakePerIndex_df.fillna(0)
                    .drop("missing_cbg", axis=1)
                    .reset_index(drop=True)
                )
    # X, y = prepare_features_target(carbIntake, df.drop("missing_cbg", axis=1).columns.values)

    return carbIntake, scaler


    

if __name__ == "__main__":
    # Load the dataset
    file = "data/Ohio2020_processed/train/540-ws-training_processed.csv"
    
    carbIntake, scaler = createDataSet(file)
    display(carbIntake[0])