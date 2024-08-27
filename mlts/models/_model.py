class Model:
    """
    Abstract model class
    """
    
    def __init__(self, **kwargs):
        pass
    
    def fit(self, data, **kwargs):
        raise NotImplementedError
    
    def predict(self, data, **kwargs):
        raise NotImplementedError
