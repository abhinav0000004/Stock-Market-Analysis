from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential, load_model
from mlts.config import ModelParams, ModelPath
from mlts.utils.data import split_data
from mlts.utils.save import save_model
from mlts.models import Model
import keras.layers as kl
import numpy as np
import keras_tuner


class LSTM(Model):
    """
    LSTM model class
    """
    
    def __init__(self):
        super().__init__()
        
        # Set random seed
        np.random.seed(42)
        
        # Model variable for prediction
        self._model = None
        
        # Model variables for training
        self._x_train_shape = None
        self._y_train_shape = None
    
    def _build_model(self, hp):
        # Define model
        model = Sequential()
        
        model.add(
            kl.LSTM(
                units=hp.Int('input_unit', min_value=32, max_value=512, step=32),
                return_sequences=True,
                input_shape=(self._x_train_shape[1], 1)
            )
        )
        
        for i in range(hp.Int('n_layers', 1, 4)):
            model.add(
                kl.LSTM(
                    hp.Int(f'lstm_{i}_units', min_value=32, max_value=512, step=32),
                    return_sequences=True
                )
            )
        
        model.add(kl.LSTM(hp.Int('layer_2_neurons', min_value=32, max_value=512, step=32)))
        model.add(kl.Dropout(hp.Float('Dropout_rate', min_value=0, max_value=0.5, step=0.1)))
        model.add(
            kl.Dense(
                self._y_train_shape[1],
                activation=hp.Choice('dense_activation', ['relu', 'sigmoid'])
            )
        )
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        return model
    
    def fit(self, df, **kwargs):
        # Split the data into training and testing sets
        train_data, test_data = split_data(df)
        
        # Create the training and testing sets
        target_var = ModelParams.TARGET.value
        input_vars = df.columns.drop(target_var)
        x_train = np.array(train_data[input_vars], dtype=np.float32)
        y_train = np.array(train_data[[target_var]], dtype=np.float32)
        
        # Reshape the data for the LSTM model
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Assign shapes
        self._x_train_shape = x_train.shape
        self._y_train_shape = y_train.shape
        
        # Model tuning
        tuner = keras_tuner.tuners.RandomSearch(
            self._build_model,
            objective='loss',
            max_trials=ModelParams.MAX_TRIALS.value,
            executions_per_trial=ModelParams.EXECUTIONS_PER_TRIAL.value,
        )
        
        tuner.search(
            x=x_train,
            y=y_train,
            epochs=ModelParams.EPOCHS.value,
            batch_size=ModelParams.BATCH_SIZE.value
        )
        
        # Tuned best model
        best_model = tuner.get_best_models(num_models=1)[0]
        self._model = best_model
        print(self._model.summary())
        
        # Best hyperparameters
        best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
        print('Best hyperparameters: ', best_hyperparameters.values)
        
        # Create the testing sets
        x_test = np.array(test_data[input_vars], dtype=np.float32)
        y_test = np.array(test_data[[target_var]], dtype=np.float32)
        
        # Evaluate Model
        results = self._model.evaluate(x_test, y_test, batch_size=1)
        print('Results: ', results)
        
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
        save_model(self._model, 'LSTM', dataset=dataset)
    
    def predict(self, data, **kwargs):
        """
        Predicts the next value in the sequence
        
        Args:
            data (object): Data to predict
            **kwargs: Keyword arguments
    
        Returns:
            object: Predicted value
        """
        
        try:
            if self._model is None:
                self._model = load_model(ModelPath.LSTM.value)
            
            if self._model:
                predictions = self._model.predict(data)
                
                return predictions
        
        except Exception as ex:
            raise Exception('Error predicting data: ', ex)
