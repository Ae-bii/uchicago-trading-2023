import pandas as pd
import numpy as np
import scipy.optimize as sco


#########################################################################
## Change this code to take in all asset price data and predictions    ##
## for one day and allocate your portfolio accordingly.                ##
#########################################################################


def portfolio_returns(weights, returns):
    return np.sum(weights*returns)
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
def portfolio_cvar(weights, returns):
    portfolio_ret = portfolio_returns(weights, returns)
    portfolio_returns_sorted = np.sort(portfolio_ret)
    cvar = np.mean(portfolio_returns_sorted[:int(len(portfolio_returns_sorted)*0.1)])
    return -cvar
def optimize_portfolio(returns):
    n = len(returns.columns)
    # Initial weights
    weights = np.random.random(n)
    weights /= np.sum(weights)
    # Constraints
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Bounds
    bounds = tuple((0,1) for i in range(n))
    # Optimization
    opt_portfolio = sco.minimize(portfolio_cvar, weights, (returns,), method='SLSQP', bounds=bounds, constraints=cons)
    return opt_portfolio
def get_results(returns):
    # Optimization
    opt_portfolio = optimize_portfolio(returns)
    # Optimal weights
    weights = opt_portfolio['x']
    # Portfolio statistics
    ret = portfolio_returns(weights, returns)
    vol = portfolio_volatility(weights, returns.cov())
    cvar = -portfolio_cvar(weights, returns)
    # Return results
    results = {
        'weights': weights,
        'returns': ret,
        'volatility': vol,
        'cvar': cvar
    }
    return results

import pandas as pd
import numpy as np
from scipy.stats import norm

def calculate_cvar(returns, alpha):
    """Calculate CVaR at alpha level."""
    sorted_returns = sorted(returns)
    index = int(alpha * len(sorted_returns))
    cvar = sum(sorted_returns[:index]) / index
    return cvar

def markowitz_cvar_portfolio(df, alpha=0.05):
    """Calculate Markowitz portfolio allocation using CVaR."""
    returns = df.pct_change().dropna()

    # Calculate mean, variance, skewness, and kurtosis of returns
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    # Set up optimization problem
    num_assets = len(df.columns)
    weights = np.ones(num_assets) / num_assets

    def portfolio_cvar(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        z = norm.ppf(alpha)
        portfolio_cvar = - portfolio_return + (z * portfolio_volatility * (1 + (skewness.dot(weights) / 6) * z - ((kurtosis.dot(weights) - 3) / 24) * (z ** 2)))
        return portfolio_cvar

    # Minimize CVaR
    from scipy.optimize import minimize
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for i in range(num_assets))
    optimized_results = minimize(portfolio_cvar, weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Calculate Sharpe ratio
    portfolio_return = np.dot(optimized_results.x, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(optimized_results.x.T, np.dot(cov_matrix, optimized_results.x)))
    sharpe_ratio = (252 ** 0.5) * (portfolio_return) / portfolio_volatility

    # Print results
    # print("Weights:", optimized_results.x)
    # print("Portfolio CVaR:", -optimized_results.fun)
    # print("Sharpe Ratio:", sharpe_ratio)
    return optimized_results.x


data = pd.read_csv("case3\data\Training Data_Case 3.csv", index_col=0)

init_df = data[:504]

def allocate_portfolio(asset_prices):
    
    # This simple strategy equally weights all assets every period
    # (called a 1/n strategy).
    global init_df
    
    init_df = pd.concat([init_df, pd.DataFrame([asset_prices], columns=init_df.columns)], ignore_index=True)
    
    # results = get_results(init_df.pct_change().dropna())
    
    
    return markowitz_cvar_portfolio(init_df)



def grading(testing): #testing is a pandas dataframe with price data, index and column names don't matter
    weights = np.full(shape=(len(testing.index),10), fill_value=0.0)
    for i in range(0,len(testing)):
        unnormed = np.array(allocate_portfolio(list(testing.iloc[i,:])))
        positive = np.absolute(unnormed)
        normed = positive/np.sum(positive)
        weights[i]=list(normed)
        print(i)
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