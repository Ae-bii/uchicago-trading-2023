import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize

#########################################################################
## Change this code to take in all asset price data and predictions    ##
## for one day and allocate your portfolio accordingly.                ##
#########################################################################

data = pd.read_csv("case3\data\Training Data_Case 3.csv", index_col=0)

def allocate_portfolio(asset_prices):
    
    # This simple strategy equally weights all assets every period
    # (called a 1/n strategy).
    
    n_assets = len(asset_prices)
    weights = np.repeat(1 / n_assets, n_assets)
    return weights



def grading(testing): #testing is a pandas dataframe with price data, index and column names don't matter
    weights = np.full(shape=(len(testing.index),10), fill_value=0.0)
    for i in range(0,len(testing)):
        unnormed = np.array(allocate_portfolio(list(testing.iloc[i,:])))
        positive = np.absolute(unnormed)
        normed = positive/np.sum(positive)
        weights[i]=list(normed)
    capital = [1]
    for i in range(len(testing) - 1):
        shares = capital[-1] * np.array(weights[i]) / np.array(testing.iloc[i,:])
        capital.append(float(np.matmul(np.reshape(shares, (1,10)),np.array(testing.iloc[i+1,:]))))
    returns = (np.array(capital[1:]) - np.array(capital[:-1]))/np.array(capital[:-1])
    return np.mean(returns)/ np.std(returns) * (252 ** 0.5), capital, weights

output = grading(data[504:])

# Beat 1.0708584866024577 which is from uniform
# 1.4792177903232782 from MaxSharpe with initial 504 datapoints
print("Sharpe Ratio: ", output[0])