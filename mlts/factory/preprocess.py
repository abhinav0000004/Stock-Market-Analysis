from mlts.preprocessor.stock import StockPreprocessor
from mlts.factory import Factory


class PreprocessorFactory(Factory):
    """
    Preprocessor Factory
    """
    
    def get(self, name, **kwargs):
        if name == 'stock':
            return StockPreprocessor()
        else:
            raise Exception(f"Preprocessor {name} not found")
