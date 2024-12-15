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