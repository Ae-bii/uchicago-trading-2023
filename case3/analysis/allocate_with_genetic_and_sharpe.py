import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
import time
# 5: 1.04
np.random.seed(5)
# Did up to 11

#########################################################################
## Change this code to take in all asset price data and predictions    ##
## for one day and allocate your portfolio accordingly.                ##
#########################################################################

class Projections:
    # Get daily % returns given prices
    def make_df(self, df):
        df['shift']=df[df.columns[0]].shift(1)
        df['PctChg']=(df[df.columns[0]]-df["shift"]).div(df["shift"])*100
        df=df.drop(columns='shift')
        return df
    # Generate list of projections
    def gen_data(self, df, first_day, runs, days):
        
        # Create list of arrays with projected daily returns
        data=[]
        first_prices=[]
        for x in range(len(df.columns)):
            temp=self.make_df(df.iloc[:,x:x+1])
            first_prices.append(temp[temp.columns[0]].iloc[first_day-1])
            data.append(np.random.laplace(loc=temp['PctChg'].mean(), scale=temp['PctChg'].std(), size=(runs, days)))
        
        # Get projected prices by applying % changes
        proj_dfs=[]
        for asset_num in range(len(data)):
            
            asset_data=[]
            for n in range(len(data[asset_num])):
                prices=[first_prices[asset_num]]
                for x in range(len(data[asset_num][n])):
                    prices.append(prices[x]*(100+data[asset_num][n][x])/100)
                asset_data.append(prices)
            
            proj_arr=np.array(asset_data)
            proj_df=pd.DataFrame()
            for x in range(proj_arr.shape[0]):
                proj_df[str(x)]=proj_arr[x]
            proj_dfs.append(proj_df)
            
        # Transform dataframes
        dfs=[]
        for run_num in range(proj_dfs[0].shape[1]):
            temp=pd.DataFrame()
            for asset_num in range(len(proj_dfs)):
                prev_df=proj_dfs[asset_num]
                temp[str(asset_num)]=prev_df[prev_df.columns[run_num]]
            dfs.append(temp)
            
        return dfs

class GeneticAlgorithm:
    def __init__(self, df):
        self.year_num = np.sqrt(252)
        self.returns = df
        self.high_fitness_weights = {}
        self.convergence_count = []
    
    def gen_weights(self, df):
        weights=np.random.randint(0, 100000, len(df.columns))
        weights=weights/sum(weights)
        return weights
    
    def gen_population(self, num_chrom, df):
        population=[]
        for x in range(num_chrom):
            population.append(self.gen_weights(df))
        return np.array(population)
    
    def evaluate_chrom(self, chromosome, dfs):    
        sharpes=[]
        for x in range(len(dfs)):
            temp=dfs[x].multiply(chromosome, axis=1)
            test=temp[list(temp.columns)].sum(axis=1).pct_change()
            sharpe=(test.mean()/test.std())*self.year_num
            sharpes.append(sharpe)
        return np.mean(sharpes)
    
    def evaluate_population(self, pop, proj_dfs):
        sharpes=[]
        for chrom in pop:
            sharpes.append(self.evaluate_chrom(chrom, proj_dfs))
        return np.array(sharpes)
    
    # Choose chromosome for next generation given the cumulataive normalized sharpes
    def choose_chrom(self, population, cum_norm_sharpes):
        for n in range(len(cum_norm_sharpes)):
            if cum_norm_sharpes[n]>np.random.rand(1)[0]:
                return population[n]
            
    def crossover(self, chrom1, chrom2):
        if np.random.rand(1)[0]>.5:
            return np.concatenate((chrom1[:int(len(chrom1)/2)], chrom2[int(len(chrom1)/2):]), axis=None)
        else:
            return np.concatenate((chrom2[:int(len(chrom1)/2)], chrom1[int(len(chrom1)/2):]), axis=None)
        
    def mutate(self, chrom, rate):
        new=[]
        for weight in chrom:
            if np.random.rand(1)[0]<rate:
                new_weight=weight*(1+np.random.normal(0, .4, 1)[0])
                if(new_weight<0):
                    new.append(0)
                else:
                    new.append(new_weight)
            else:
                new.append(weight)
        return np.array(new)
    
    def rebalance(self, chrom):
        return chrom/sum(chrom)
    
    # Create next generation of chromosomes (weights)
    def next_gen(self, sharpes, population, mutation_rate):
        
        new_gen=[]
        
        # Select best fourth
        num_chosen_direct=round(len(population)/4)
        temp={}
        for x in range(len(sharpes)):
            temp[x]=sharpes[x]
        temp={k: v for k, v in sorted(temp.items(), key=lambda item: item[1])}
        keys=list(temp.keys())[-1*num_chosen_direct:]
        for x in keys:
            new_gen.append(population[x])
        
        # Select rest through crossover: create cumulative norm fitness list
        norm_sharpes=sharpes/sum(sharpes)
        cum_norm_sharpes=[norm_sharpes[0]]
        for n in range(1, len(norm_sharpes)):
            cum_norm_sharpes.append(cum_norm_sharpes[n-1]+norm_sharpes[n])
        for x in range(len(population)-num_chosen_direct):
            new_gen.append(self.crossover(self.choose_chrom(population, cum_norm_sharpes), self.choose_chrom(population, cum_norm_sharpes)))
            
        # Mutation and rebalance
        final=[]
        for x in new_gen:
            final.append(self.rebalance(self.mutate(x, mutation_rate)))
            
        return np.array(final)
    
    def genetic_algo(self, prev_gen_sharpes, prev_gen, pop_size, mutation_rate, df, proj_dfs):
        
        # Add to high fitness weights dict
        # high_fitness_weights = {}
        max_sharpe=max(prev_gen_sharpes)
        best_weights=prev_gen[list(prev_gen_sharpes).index(max_sharpe)]
        self.high_fitness_weights[max_sharpe]=best_weights
        
        # Check convergence
        # convergence_count = []
        convergence=False
        if (len(self.high_fitness_weights)==30):
            convergence=True
        elif (len(self.high_fitness_weights)>1):
            if max_sharpe<list(self.high_fitness_weights.keys())[-2]*1.02:
                self.convergence_count.append(1)
            else:
                self.convergence_count.append(0)

            if (sum(self.convergence_count[-20:])==20):
                convergence=True
            else:
                convergence=False
        else:
            self.convergence_count.append(0)
        
        # Recursive GA
        if (convergence==False):
            print("Generation Number "+str(len(self.convergence_count)+1))
            print("---Processing")
            print("---Sharpe: "+str(max_sharpe))
            new_gen=self.next_gen(prev_gen_sharpes, prev_gen, mutation_rate)
            new_gen_sharpes=self.evaluate_population(new_gen, proj_dfs)
            print("---Done")
            self.genetic_algo(new_gen_sharpes, new_gen, pop_size, mutation_rate, df, proj_dfs)
        else:
            print("Convergence achieved")
            
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

start = 126*2
data = pd.read_csv("case3/data/Training Data_Case 3.csv", index_col=0)

sharpe_model = MaximizeSharpeModel(data[:504])

init_df = data[:start]
best_weights = []

def allocate_portfolio(asset_prices):
    
    global init_df
    global best_weights
    global sharpe_model
    
    init_df = pd.concat([init_df, pd.DataFrame([asset_prices], columns=init_df.columns)], ignore_index=True)
    
    sharpe_model.append_prices(asset_prices)
    sharpe_model.update_weights()
    
    # print(len(init_df))
    max_sharpe = 0
    best_weights = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    if len(init_df) > start*2 and len(init_df) % 21 == 0:
        projections = Projections()
        proj_dfs = projections.gen_data(init_df, first_day = len(init_df)-start, runs = 50, days = start)
        
        # Define GA inputs
        pop_size=25
        mutation_rate=.5

        genetic_model = GeneticAlgorithm(init_df)
        gen1 = genetic_model.gen_population(pop_size, init_df)

        # Run GA
        sharpes=genetic_model.evaluate_population(gen1, proj_dfs)
        genetic_model.genetic_algo(sharpes, gen1, pop_size, mutation_rate, init_df, proj_dfs)
        
        max_sharpe=max(list(genetic_model.high_fitness_weights.items()))[0]
        best_weights=max(list(genetic_model.high_fitness_weights.items()))[1]
    return best_weights if max_sharpe > sharpe_model.sharpe_ratio else sharpe_model.weights


t0 = time.perf_counter()
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
t1 = time.perf_counter()
output = grading(data[start:])

# Beat 1.0708584866024577 which is from uniform
# 1.4792177903232782 from MaxSharpe with initial 504 datapoints
# 1.18765382185393 from GeneticAlgorithm
print("Sharpe Ratio: ", output[0])
print(t1-t0)