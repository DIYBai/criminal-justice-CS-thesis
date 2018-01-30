import pandas as pd

def parse_data(filename="RiskAssessData.csv"):
    riskAsses = pd.read_csv(filename, header = 0)
    original_headers = list(riskAsses.columns.values)
    riskAsses = riskAsses._get_numeric_data()
    RA_array = riskAsses.as_matrix()
    return RA_array[:,4:33], RA_array[:,37]
