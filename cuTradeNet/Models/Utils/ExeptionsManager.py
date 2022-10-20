'''Here im going to place Exception checking functions to be used in the models'''
import numpy as np

def check_MCS(MCS : int):
    '''Check if the MCS is a valid number'''
    if MCS<=0:
        raise Exception('MCS must be a positive integer')
    
def check_wealths(wealths : list):
    '''Check if the wealths are valid'''
    if np.any(np.array(wealths)<0):
        raise Exception('All wealths must be positive')

def check_wmin(wmin : float):
    '''Check if the wmin is valid'''
    if wmin<0:
        raise Exception('Minimum wealth to transact must be positive')

def check_risks(risks):
    '''Check if the risks are valid'''
    risks=np.asarray(risks)
    if np.any(risks>1) or np.any(risks<0):
        raise Exception('All risks must be between 0 and 1')
        