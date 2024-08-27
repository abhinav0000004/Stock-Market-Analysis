from mlts.config import ModelPath, PreprocessedDataset
import os.path
import pickle


def save_model(model, name, **kwargs):
    """
    Save the model to disk
    
    Args:
        model (object): Model object
        name (str): Name of the model
        **kwargs: Keyword arguments
    """
    # Model path
    model_path = ModelPath[name].value
    
    if not os.path.isdir(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    
    if name in ['LSTM']:
        if kwargs.get('dataset') is not None:
            model_path = model_path.replace('.h5', f"_{(kwargs.get('dataset')).lower()}.h5")
        
        model.save(
            model_path,
            overwrite=True,
            include_optimizer=True,
            save_format='h5',
            save_traces=True,
        )
    
    elif name in ['ARIMA', 'XGB']:
        if kwargs.get('dataset') is not None:
            model_path = model_path.replace('.pkl', f"_{(kwargs.get('dataset')).lower()}.pkl")
        
        with open(model_path, 'wb') as pkl:
            pickle.dump(model, pkl)
    else:
        raise ValueError('Unknown model name: {}'.format(name))


def load_model_files(name, **kwargs):
    """
    Save the model to disk

    Args:
        name (str): Name of the model
        **kwargs: Keyword arguments
    """
    # Model path
    model_path = ModelPath[name].value
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if name in ['ARIMA', 'XGB']:
        if kwargs.get('dataset') is not None:
            model_path = model_path.replace('.pkl', f"_{(kwargs.get('dataset')).lower()}.pkl")
        
        with open(model_path, 'rb') as pkl:
            return pickle.load(pkl)
    else:
        raise ValueError('Cannot load unknown model: {}'.format(name))


def save_data(df, name, **kwargs):
    """
    Save the data to disk
    
    Args:
        df (pd.DataFrame): Dataframe to save
        name (str): Name of the data
        **kwargs: Keyword arguments
    """
    
    try:
        # Dataframe disk path
        dataframe_path = PreprocessedDataset[name].value
        
        if not os.path.isdir(os.path.dirname(dataframe_path)):
            os.makedirs(os.path.dirname(dataframe_path))
        
        df.to_csv(dataframe_path)
    
    except Exception as ex:
        raise Exception(f"Dataframe saving failed: {ex}")
