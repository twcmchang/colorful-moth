import math
import numpy as np

class Config(object):
    """Base configuration."""
    NAME = None  # Override in sub-classes
    
    KATIE = 'awesome'

    
    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = 16
        self.BATCH_SIZE1=  12

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
