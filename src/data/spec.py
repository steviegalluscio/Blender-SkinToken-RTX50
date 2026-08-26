from abc import ABC, abstractmethod
from dataclasses import fields

class ConfigSpec(ABC):
    @classmethod
    def check_keys(cls, config, expect=None):
        # Gracefully allow any checkpoint metadata keys during inference
        pass
    
    @classmethod
    @abstractmethod
    def parse(cls, **kwargs) -> 'ConfigSpec':
        raise NotImplementedError()