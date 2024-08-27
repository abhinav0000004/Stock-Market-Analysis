import pandas as pd


def split_data(data):
    """
    Split data into training and testing sets
    """
    train_data = data[:int(len(data) * 0.8)]
    test_data = data[int(len(data) * 0.8):]
    return train_data, test_data


def split_date(df, target_col, new_alias=None):
    """
    Utility to convert a date column into various date entities like year, month, day, etc.

    Args:
        df (pd.DataFrame): Input DataFrame
        target_col (string): Feature on which the split needs to be performed. Must be of datetime type
        new_alias (string): String alias to be used for split columns generated from target_col.
                            In case no value is passed then it is same as target_col
    """
    
    if new_alias is None:
        new_alias = target_col
    
    df[f'{new_alias}_year'] = df[target_col].dt.year
    df[f'{new_alias}_month'] = df[target_col].dt.month
    df[f'{new_alias}_day'] = df[target_col].dt.day
    df[f'{new_alias}_quarter'] = df[target_col].dt.quarter
    df[f'{new_alias}_is_month_start'] = df[target_col].dt.is_month_start.astype(int)
    df[f'{new_alias}_is_month_end'] = df[target_col].dt.is_month_end.astype(int)
    
    return df


def moving_average(df, feature, n_days):
    """
    Function to find out Moving Average of certain column for past N days
    
    Args:
        df (pd.DataFrame): Input Dataframe
        feature (str):
        n_days (int): Number of past days to consider for moving average

    Returns:
        df (pd.DataFrame): Dataframe with added column of moving Average.
    """
    df[feature + '_mean'] = df[feature].rolling(window=n_days, min_periods=1).mean()
    df[feature + '_std'] = df[feature].rolling(window=n_days, min_periods=1).std()
    
    return df


def enrich_stock_features(df, num_days=5):
    """
    Function to do Feature Engineering on stock data
        - Difference between high & low
        - Difference between open & low and order day.
        - Creating lag window columns
        - Adding Moving average column for features.
        - Dropping unnecessary columns

    Args:
        df (pd.DataFrame): Input Dataframe
        num_days (int): Number of days to consider for lag window

    Returns:
        df (pd.DataFrame): Dataframe with enriched features
    """
    
    try:
        # Returns
        df['returns_prc'] = 100 * ((df['close'] - df['open']) / df['close'])
        
        # High-Low difference
        df['high_low_diff'] = df['high'] - df['low']
        
        # Open-CLose Difference
        df['open_close_diff'] = df['open'] - df['close']
        
        # Order Day
        df['day_num'] = [x for x in list(range(len(df)))]
        
        # Features To Create Lag
        lag_features = ['high_low_diff', 'open_close_diff', 'volume', 'adj_close']
        
        # Adding Lag Columns
        lag_range = [x + 1 for x in range(num_days)]
        merging_keys = ['day_num']
        
        for val in lag_range:
            temp_df = df[merging_keys + lag_features].copy()
            temp_df['day_num'] = temp_df['day_num'] + val
            new_feature_alias = lambda x: 'lag_{}_{}'.format(val, x) if x in lag_features else x
            temp_df = temp_df.rename(columns=new_feature_alias)
            df = pd.merge(df, temp_df, on=merging_keys, how='left')
        
        # Remove the first N rows since it doesn't contain any NaN values
        df = df[num_days:]
        
        # Adding moving average column for each feature
        for val in lag_features:
            df = moving_average(df, val, num_days)
        
        return df
    
    except Exception as ex:
        raise Exception(f"Feature Engineering Failed {ex}")


def normalize(row, mean, std):
    """
    This function is used to normalize the data.
    
    Args:
        row: Dataframe row
        mean: Mean of the row
        std: Standard deviation of the row

    Returns:
        norm_row: Normalized row
    """
    
    std = max(0.001, std)
    norm_row = (row - mean) / std
    return norm_row


def scale_stocks_data(df):
    """
    Function to scale the data using mean and standard deviation of each feature
    
    Args:
        df (pd.DataFrame): Input Dataframe

    Returns:
        df (pd.DataFrame): Scaled Dataframe
    """
    
    try:
        features_scaled = ['high_low_diff', 'open_close_diff', 'volume', 'adj_close']
        
        for feature in features_scaled:
            feat_list = [feature, 'lag_1_' + feature, 'lag_2_' + feature, 'lag_3_' + feature]
            temp = df.apply(
                lambda row: normalize(row[feat_list], row[feature + '_mean'], row[feature + '_std']),
                axis=1
            )
            df = pd.concat([df.drop(columns=feat_list), temp], axis=1)
        
        return df
    
    except Exception as ex:
        raise Exception(f"Scaling Failed {ex}")
