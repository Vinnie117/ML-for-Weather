############ File for custom functions ############

#### Custom function for adjusted R2

def adjustedR2(r2, data):

    adjustedR2 =  1 - (1 - r2) * ((data.shape[0] - 1) / (data.shape[0] - data.shape[1] - 1))
    return adjustedR2
















