from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
from mlts.utils.data import split_data
from mlts.utils.save import save_model
from pmdarima.arima import auto_arima
from mlts.config import ModelParams
from mlts.models import Model
from tqdm import tqdm
import numpy as np


class ARIMA(Model):
    """
    ARIMA model class
    """
    
    def __init__(self):
        super().__init__()
        self._model = None
        np.random.seed(42)
    
    def fit(self, data, **kwargs):
        # Split the data into training and testing sets
        train_data, test_data = split_data(data)
        
        # Build the LSTM model
        self._model = auto_arima(
            train_data['adj_close'],
            start_p=0,
            start_q=0,
            test='adf',  # use adftest to find optimal 'd'
            max_p=5,
            max_q=5,  # maximum p and q
            m=1,  # frequency of series
            d=None,  # let model determine 'd'
            seasonal=False,  # No Seasonality
            start_P=0,
            D=0,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        p, d, q = self._model.get_params()['order']
        print(self._model.summary())
        print('pdq values: ', p, d, q)
        
        # Compile and train the self._model
        train = train_data[ModelParams.TARGET.value].values
        test = test_data[ModelParams.TARGET.value].values
        
        # Make predictions on the test data
        history = [x for x in train]
        predictions = list()
        
        for i in tqdm(range(len(test))):
            model = StatsARIMA(history, order=(p, d, q))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[i]
            history.append(obs)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(test, predictions))
        print('Root mean squared error: ', rmse)
        
        # Mean Absolute Error
        mae = mean_absolute_error(test, predictions)
        print('mean absolute error: ', mae)
        
        # Save the model
        dataset = kwargs.get('dataset', None)
        save_model(self._model, 'ARIMA', dataset=dataset)
    
    def predict(self, data, **kwargs):
        """
        Predict the data

        Args:
            data:
            **kwargs:

        Returns:
            predictions (np.array): Predictions
        """
        if self._model is None:
            raise Exception('Model not trained')
        
        return self._model.predict(data)
