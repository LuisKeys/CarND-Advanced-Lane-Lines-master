import numpy as np

def save_float_tensor(filename, array):
    np.savetxt(filename, array, fmt='%d')

def load_float_tensor(filename):
    return np.loadtxt(filename, dtype=np.float32)
