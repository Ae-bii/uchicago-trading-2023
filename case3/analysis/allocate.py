import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize

#########################################################################
## Change this code to take in all asset price data and predictions    ##
## for one day and allocate your portfolio accordingly.                ##
#########################################################################

data = pd.read_csv("Training Data_Case 3.csv", index_col=0)

class MaximizeSharpeModel:
    def __init__(self, historical_prices):
        self.historical_prices = historical_prices
        self.num_assets = len(self.historical_prices.columns)
        
        self.returns = np.array(self.historical_prices.pct_change().dropna())
        self.mean_returns = np.mean(self.returns, axis=0)
        self.cov_matrix = np.cov(self.returns.T)
        
        self.skewness = np.array([scipy.stats.skew(ret) for ret in self.returns.T])
        self.kurtosis = np.array([scipy.stats.kurtosis(ret) for ret in self.returns.T])
        
        self.sharpe_ratio = None
        self.weights = self._optimize_weights()
        
    def _negative_sharpe_ratio(self, weights, mean_returns, cov_matrix, skewness, kurtosis):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        a = 1 + skewness @ weights + 0.5 * kurtosis @ np.square(weights)
        sharpe_ratio = (portfolio_return / portfolio_std_dev) * np.power(np.sign(a) * np.abs(a), 1/3)
        return -sharpe_ratio
    
    def _optimize_weights(self):
        # set the initial weight allocation to be equal for all assets
        weights_0 = np.array([1/self.num_assets]*self.num_assets)

        # define the constraints for the optimization
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # define the bounds for the optimization (between 0 and 1 for each weight)
        bounds = tuple((0,1) for _ in range(self.num_assets))

        # minimize the negative Sharpe ratio to find the optimal weights
        result = minimize(self._negative_sharpe_ratio, weights_0, args=(self.mean_returns, self.cov_matrix, self.skewness, self.kurtosis), method='SLSQP', bounds=bounds, constraints=constraints)

        # calculate the Sharpe ratio for the optimal weights
        optimal_weights = result.x
        optimal_return = np.dot(optimal_weights, self.mean_returns)
        optimal_std_dev = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
        self.sharpe_ratio = np.sqrt(252) * optimal_return / optimal_std_dev
        
        return result.x
    
    def append_prices(self, current_prices):
        self.historical_prices.loc[-1] = current_prices  # adding a row
        # self.historical_prices = pd.concat([self.historical_prices, pd.DataFrame(current_prices, columns=['A','B','C','D','E','F','G','H','I','J'])])
        self.returns = np.array(self.historical_prices.pct_change().dropna())
    
    def update_weights(self):
        self.mean_returns = np.mean(self.returns, axis=0)
        self.cov_matrix = np.cov(self.returns.T)
        
        self.skewness = np.array([scipy.stats.skew(ret) for ret in self.returns.T])
        self.kurtosis = np.array([scipy.stats.kurtosis(ret) for ret in self.returns.T])
        self.weights = self._optimize_weights()
        
model = MaximizeSharpeModel(data)

def allocate_portfolio(asset_prices):
    
    model.append_prices(asset_prices)
    model.update_weights()
    
    return model.weights


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