"""
Module for precomputing or hard-coding F and R symbols for different anyon types.
"""

import numpy as np

def fibonacci_f_symbol(a, b, c):
    """
    Compute the F-symbol for Fibonacci anyons.
    
    Parameters:
    -----------
    a, b, c : str
        Fusion channel labels ("1" or "tau")
        
    Returns:
    --------
    F : float or complex
        The F-symbol value
    """
    if a == b == c == "1":
        return 1.0
    elif a == b == c == "tau":
        return 1.0 / np.sqrt(1 + np.sqrt(5))
    elif a == b == "tau" and c == "1":
        return np.sqrt(1 + np.sqrt(5))
    else:
        return 0.0

def fibonacci_r_symbol(a, b):
    """
    Compute the R-symbol for Fibonacci anyons.
    
    Parameters:
    -----------
    a, b : str
        Fusion channel labels ("1" or "tau")
        
    Returns:
    --------
    R : float or complex
        The R-symbol value
    """
    if a == b == "1":
        return 1.0
    elif a == b == "tau":
        return np.exp(1j * np.pi / 5)
    else:
        return 0.0

def ising_f_symbol(a, b, c):
    """
    Compute the F-symbol for Ising anyons.
    
    Parameters:
    -----------
    a, b, c : str
        Fusion channel labels ("1" or "psi")
        
    Returns:
    --------
    F : float or complex
        The F-symbol value
    """
    if a == b == c == "1":
        return 1.0
    elif a == b == c == "psi":
        return 1.0
    elif a == b == "psi" and c == "1":
        return 1.0
    else:
        return 0.0

def ising_r_symbol(a, b):
    """
    Compute the R-symbol for Ising anyons.
    
    Parameters:
    -----------
    a, b : str
        Fusion channel labels ("1" or "psi")
        
    Returns:
    --------
    R : float or complex
        The R-symbol value
    """
    if a == b == "1":
        return 1.0
    elif a == b == "psi":
        return -1.0
    else:
        return 0.0

def majorana_f_symbol(a, b, c):
    """
    Compute the F-symbol for Majorana anyons.
    
    Parameters:
    -----------
    a, b, c : str
        Fusion channel labels ("1" or "psi")
        
    Returns:
    --------
    F : float or complex
        The F-symbol value
    """
    return ising_f_symbol(a, b, c)

def majorana_r_symbol(a, b):
    """
    Compute the R-symbol for Majorana anyons.
    
    Parameters:
    -----------
    a, b : str
        Fusion channel labels ("1" or "psi")
        
    Returns:
    --------
    R : float or complex
        The R-symbol value
    """
    return ising_r_symbol(a, b)
