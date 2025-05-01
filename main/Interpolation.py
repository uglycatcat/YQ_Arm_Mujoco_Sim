import numpy as np
import keyboard
import time
import math
from scipy.spatial.transform import Rotation as R

class InterpolationMethod:
    def __init__(self):
        pre_data_length =5
        after_data_length =20