from mlts.config import RawDataset
from mlts.factory import Factory
import pandas as pd


class DatasetFactory(Factory):
    """
    Dataset Factory
    """
    
    def get(self, name, *args, **kwargs):
        return pd.read_csv(RawDataset[name].value, parse_dates=RawDataset.DATE_FEATURES.value)
