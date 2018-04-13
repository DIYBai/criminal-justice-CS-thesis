import pandas as pd

##  returns matricies for inputs and outputs of the file  ##
def parse_data(filename="RiskAssessData.csv", firstInpCol = 4, lastInpCol = 33, outCol = -1):
    riskAsses = pd.read_csv(filename, header = 0)
    riskAsses = riskAsses._get_numeric_data()
    RA_array = riskAsses.as_matrix()
    if outCol == -1:
        outCol = len(RA_array[0]) - 1
    return RA_array[:, firstInpCol:lastInpCol], RA_array[:, outCol]
