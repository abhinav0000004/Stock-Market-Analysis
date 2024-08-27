from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from mlts.utils.data import split_data
from mlts.utils.save import save_model
from mlts.config import ModelParams
from xgboost import XGBRegressor
from mlts.models import Model
import pandas as pd
import numpy as np


class XGB(Model):
    """
    XGBoost Model
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self._model = None
    
    def fit(self, df, **kwargs):
        """
        Fit the model
        
        Args:
            df (pd.DataFrame): Data to fit the model
            **kwargs: Additional arguments
        """
        train_data, test_data = split_data(df)
        
        # Create the training and testing sets
        target_var = ModelParams.TARGET.value
        input_vars = df.columns.drop(target_var)
        x_train = np.array(train_data[input_vars], dtype=np.float32)
        y_train = np.array(train_data[[target_var]], dtype=np.float32)
        
        # Xgboost Model
        estimator = XGBRegressor(seed=42)
        
        # Grid search cross validation to get optimal parameters
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=ModelParams.XGB_PARAMS.value,
            scoring='neg_mean_squared_error',
        )
        self._model = grid_search.fit(x_train, y_train)
        
        print('Best parameters: ', self._model.best_params_)
        print('Best score: ', self._model.best_score_)
        
        # Create the testing sets
        x_test = np.array(test_data[input_vars], dtype=np.float32)
        y_test = np.array(test_data[[target_var]], dtype=np.float32)
        
        # Predictions
        predictions = self._model.predict(x_test)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print('Root mean squared error: ', rmse)
        
        # Mean Absolute Error
        mae = mean_absolute_error(y_test, predictions)
        print('mean absolute error: ', mae)
        
        # Save the model
        dataset = kwargs.get('dataset', None)
        save_model(self._model, 'XGB', dataset=dataset)
    
    def predict(self, data, **kwargs):
        """
        Predict the data

        Args:
            data (object): Data to predict
            **kwargs: Keyword arguments
    
        Returns:
            object: Predicted value
        """
        
        if self._model is None:
            raise Exception('Model not trained')
        
        return self._model.predict(data)
