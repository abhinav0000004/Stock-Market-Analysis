from datetime import datetime as dt
from enum import Enum
import os


class Path(Enum):
    """
    Path
    """
    
    ROOT = os.path.abspath('./mlts')


class RawDataset(Enum):
    """
    Original Dataset
    """
    
    DATE_FEATURES = ['Date']
    AAPL = os.path.join(Path.ROOT.value, 'static/datasets/original/aapl.csv')
    GMBL = os.path.join(Path.ROOT.value, 'static/datasets/original/gmbl.csv')
    TSLA = os.path.join(Path.ROOT.value, 'static/datasets/original/tsla.csv')


class Preprocess(Enum):
    """
    Preprocessing Config
    """
    
    DROP_FEATURES = [
        'open', 'high', 'low', 'close',
        'day_num', 'volume_mean', 'volume_std',
        'adj_close_mean', 'adj_close_std'
    ]
    NUM_DAYS = 3  # After several iterations, 3 days is the best


class PreprocessedDataset(Enum):
    """
    Dataset Paths
    """
    
    AAPL = os.path.join(Path.ROOT.value, 'static/datasets/preprocessed_aapl.csv')
    GMBL = os.path.join(Path.ROOT.value, 'static/datasets/preprocessed_gmbl.csv')
    TSLA = os.path.join(Path.ROOT.value, 'static/datasets/preprocessed_tsla.csv')


class ModelPath(Enum):
    """
    Model Paths
    """
    XGB = os.path.join(Path.ROOT.value, f"static/checkpoints/xgb/{dt.now().strftime('%Y%m%d_%H%M_')}xgb.pkl")
    LSTM = os.path.join(Path.ROOT.value, f"static/checkpoints/lstm/{dt.now().strftime('%Y%m%d_%H%M_')}lstm.h5")
    ARIMA = os.path.join(Path.ROOT.value, f"static/checkpoints/arima/{dt.now().strftime('%Y%m%d_%H%M_')}arima.pkl")


class ModelParams(Enum):
    """
    Model Parameters
    """
    
    TARGET = 'adj_close'
    EPOCHS = 20
    VERBOSE = 2
    BATCH_SIZE = 32
    
    # For LSTM
    MAX_TRIALS = 5
    EXECUTIONS_PER_TRIAL = 3
    
    # For XGB
    XGB_PARAMS = {
        # max_depth :Maximum tree depth for base learners
        'max_depth': range(2, 10, 2),
        
        # n_estimators: Number of boosted trees to fit
        'n_estimators': range(20, 200, 30),
        
        # learning_rate:Boosting learning rate (xgb’s “eta”)
        'learning_rate': [0.001, 0.01, 0.1],
        
        # subsample : Subsample ratio of the training instance
        'subsample': [0, 0.25, 0.5, 0.75, 1],
        
        # gamma : Minimum loss reduction required to make a further partition on a leaf node of the tree
        'gamma': [0, 0.2, 0.4, 0.6, 0.8],
        
        # min_child_weight : Minimum sum of instance weight(hessian) needed in a child
        'min_child_weight': range(1, 21, 2),
        
        # colsample_bytree :Subsample ratio of columns when constructing each tree
        'colsample_bytree': [0, 0.25, 0.5, 0.75, 1],
        
        # colsample_bylevel :Subsample ratio of columns for each split, in each level
        'colsample_bylevel': [0.5, 0.75, 1]
    }
