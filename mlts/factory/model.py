from mlts.models import LSTM, XGB, ARIMA
from mlts.factory import Factory


class ModelFactory(Factory):
    """
    Model Factory
    """
    
    def get(self, name, *args, **kwargs):
        
        if name == 'LSTM':
            return LSTM(*args, **kwargs)
        elif name == 'XGB':
            return XGB(*args, **kwargs)
        elif name == 'ARIMA':
            return ARIMA(*args, **kwargs)
        else:
            raise ValueError('Unknown model name: {}'.format(name))
