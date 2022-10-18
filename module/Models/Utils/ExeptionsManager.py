'''Here im goin to place Exception checking functions to be used in the models'''



def check_MCS(MCS : int):
    '''Check if the MCS is a valid number'''
    if MCS<=0:
        raise Exception('MCS must be a positive integer')
    