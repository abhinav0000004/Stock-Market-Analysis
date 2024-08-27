from mlts.utils.data import split_date, enrich_stock_features, scale_stocks_data
from mlts.preprocessor import Preprocessor
from mlts.utils.save import save_data
from mlts.config import Preprocess


class StockPreprocessor(Preprocessor):
    """
    Stock Preprocessor
    """
    
    def __init__(self):
        super().__init__()
        self._std = None
        self._mean = None
    
    def preprocess(self, df, **kwargs):
        """
        Preprocess the data
        
        Args:
            df (pd.DataFrame): Dataframe to preprocess
            kwargs: Keyword arguments
            
        Returns:
            df (pd.DataFrame): Preprocessed dataframe
        """
        
        try:
            """Column Name Formatting"""
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            """Split date column into various date entities"""
            # year, month, day, quarter, is_month_start, is_month_end
            df = split_date(df, target_col='date')
            
            """Feature Engineering"""
            df = enrich_stock_features(df, num_days=Preprocess.NUM_DAYS.value)
            
            # Assign the mean and std of the target variable
            self._std = df['adj_close_std']
            self._mean = df['adj_close_mean']
            
            # Scale Data
            df = scale_stocks_data(df)
            
            # Fill NaN values with 0
            df = df.fillna(0)
            
            # Set date as index
            df = df.reset_index(drop=True)
            df = df.set_index('date')
            
            # Round float values to 3 decimal places
            float_cols = df.select_dtypes(include=['float']).columns.tolist()
            df[float_cols] = df[float_cols].round(3)
            
            # Drop features
            df = df.drop(columns=Preprocess.DROP_FEATURES.value, errors='ignore')
            
            # Parse the keyword arguments
            #  save (bool): Save the preprocessed data to disk
            #  dataset (str): Name of the dataset
            save = kwargs.get('save', False)
            dataset = kwargs.get('dataset', None)
            
            # Save the preprocessed data to disk
            if save:
                save_data(df, dataset)
            
            return df
        
        except Exception as ex:
            raise Exception(f"Preprocessing Failed {ex}")
    
    def descale_predictions(self, predictions):
        """
        Method to reverse the scaling applied on the predictions
        
        Args:
            predictions (object): Model predictions

        Returns:
            descaled_predictions (object): Descaled predictions
        """
        
        try:
            descaled_predictions = predictions * self._std + self._mean
            return descaled_predictions
        
        except Exception as ex:
            raise Exception(f"Descaling of predictions Failed {ex}")
