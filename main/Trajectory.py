import numpy as np
import math

class ArmMotionTrajectory:
    
    def __init__(self):
        self.pre_data_length =5
        self.processed_data_length =20
        self.sampling_buffer=np.empty((0, 3), dtype=np.float32)
        
    def linear_interpolation(self):

        return None
    
    def bessel_interpolation(self):
        pass
        
    def add_data(self):
        pass
        
    def delete_data(self):
        pass
        
    def change_data(self):
        pass
        
        return None
    
trajectory = ArmMotionTrajectory()