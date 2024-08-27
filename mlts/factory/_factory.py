class Factory:
    
    def __init__(self):
        pass
    
    def get(self, name, *args, **kwargs):
        raise NotImplementedError
