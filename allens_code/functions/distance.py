import numpy as np

def sq_Euclidean_d(x, y):
    """
    Fast squared Euclidean distance using NumPy.
    
    Args:
        x (np.ndarray): First 1D array.
        y (np.ndarray): Second 1D array.
    
    Returns:
        float: Squared Euclidean distance.
    """
    return np.dot(x - y, x - y)