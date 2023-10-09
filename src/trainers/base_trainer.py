from abc import ABCMeta,abstractmethod
import json

class BaseTrainer(metaclass=ABCMeta):
    @abstractmethod
    def step(self,data_batch):
        """Takes a data sample and returns train metrics
        """
        pass

    @abstractmethod
    def __init__(self,args,config):
        """Inits dataset with config
        """
        pass

    def load_config_file(self,config_path):
        with open(config_path) as f:
            config=json.loads(f.read())
        self.config=config
        
    
    def get_config(self):
        return self.config