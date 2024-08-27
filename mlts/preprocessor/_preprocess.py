class Preprocessor:
    """
    Preprocessor
    
    An abstract class for doing preprocessing on data before modeling.
    """
    
    def __init__(self):
        pass
    
    def preprocess(self, df, **kwargs):
        raise NotImplementedError
