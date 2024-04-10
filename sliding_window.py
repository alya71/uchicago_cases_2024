import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.optimize as minimize
import os


data = pd.read_csv('Case 2 Data 2024.csv', index_col = 0)

'''
We recommend that you change your train and test split
'''

TRAIN, TEST = train_test_split(data, test_size = 0.5, shuffle = False)


class Allocator():
    def __init__(self, train_data):
        '''
        Anything data you want to store between days must be stored in a class field
        '''
        
        self.running_price_paths = train_data.copy()
        
        self.train_data = train_data.copy()

        self.past_weights = np.array([1/6 for _ in range(6)])

        self.lower_bound = -1
        self.upper_bound = 1
        self.window_size = 50
        self.method = 'SLSQP'
        
        # Do any preprocessing here -- do not touch running_price_paths, it will store the price path up to that data
        
        
    def allocate_portfolio(self, asset_prices):
        '''
        asset_prices: np array of length 6, prices of the 6 assets on a particular day
        weights: np array of length 6, portfolio allocation for the next day
        '''

        self.running_price_paths._append(asset_prices, ignore_index = True)
        
        self.train_data = self.train_data._append(pd.Series(asset_prices, index = self.train_data.columns), ignore_index = True)
        #Print the last row of the self.train_data dataframe to see if the data is being appended correctly
        #print(self.train_data.iloc[-1])


        def portfolio_var(weights,cov):
            """
            Input: Portfolio weights and covariance matrix for assets
            Return: Standard deviation of the portfolio
            """
            var = weights.T @ cov @ weights

            return var
        
        def calc_diversification_ratio(w, inp_data):
            # average weighted vol

            returns = inp_data/inp_data.shift(1)
            returns = returns.dropna()
            V = returns.cov()
            w_vol = np.dot(np.sqrt(np.diag(V)), w.T)
            # portfolio vol
            port_vol = np.sqrt(portfolio_var(w, V))
            diversification_ratio = w_vol/port_vol
            # return negative for minimization problem (maximize = minimize -)
            return -diversification_ratio
        
        def time_weighted(inp):
            #Implement basic distributed lag
            n = self.window_size
            total_sum = n * (n+1)/2
            # print(inp.shape)

            #Note: Make weighted_array a global
            weighted_array = np.array([i/total_sum for i in range(1,n+1)]).reshape(-1,1)
            # print(weighted_array.shape)
            return inp * weighted_array
        
        def neg_sharpe(weights, inp_data):

            # inp_data = time_weighted(inp_data)
            capital = [1]
            for i in range(len(inp_data) - 1): 
                shares = capital[-1] * weights / np.array(inp_data.iloc[i,:]) 
                balance = capital[-1] - np.dot(shares, np.array(inp_data.iloc[i,:]))
                net_change = np.dot(shares, np.array(inp_data.iloc[i+1,:]))
                capital.append(balance + net_change)

                # capital.append(capital[-1] * np.dot((1 + weights),(inp_data.iloc[i+1,:])/inp_data.iloc[i,:] - 1))
            capital = np.array(capital)
            returns = (capital[1:] - capital[:-1]) / capital[:-1]
            
            if np.std(returns) != 0:
                sharpe = np.mean(returns) / np.std(returns)
            else:
                sharpe = 0

            return -sharpe
        
        constrains = {'type':'eq', 'fun': lambda weights : np.sum(weights) - 1}
        
        bounds = [(self.lower_bound,self.upper_bound) for _ in range(6)] #Ensures no short selling and prevents overweighting of any asset
        
        init_weights = self.past_weights #Initial weights are equal

        #weights = minimize.minimize(neg_sharpe, init_weights, args = (log_returns, cov_matrix, 0), method = 'SLSQP', bounds = bounds, constraints = constrains)
        five_day_avg = self.train_data.tail(5).mean(axis = 0)

        ten_day_avg = self.train_data.tail(10).mean(axis = 0)

        # If five_day_avg is less than ten_day_avg, we are in a downtrend. For each of the six columns, 
        # if the five_day_avg is less than the ten_day_avg, we will set the lower bound to -.5 and the 
        # upper bound to 0.5. This will make the optimizer more conservative with the stocks
        for i in range(6):
            if five_day_avg.iloc[i] < ten_day_avg.iloc[i]:
                print('Here')
                bounds[i] = (-0.4, 0.4)

        weights = minimize.minimize(neg_sharpe, init_weights, args = (self.train_data.tail(self.window_size)), method = self.method, bounds = bounds, constraints = constrains) 
        ### TODO Have a conditional bounding algorithm based on if a downtrend is detected...be more conservative with stocks when downtrend is detected. Also weighting more recent events higher could work.
        #print(weights.x)

        self.past_weights = weights.x #Store the weights for the next day

        return weights.x
        


def grading(train_data, test_data): 
    '''
    Grading Script
    '''

    weights = np.full(shape=(len(test_data.index),6), fill_value=0.0)
    alloc = Allocator(train_data)


    #Calculates weights for each day using our allocator, passing in one test row at a time
    for i in range(0,len(test_data)):
        weights[i,:] = alloc.allocate_portfolio(test_data.iloc[i,:])
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")
    
    capital = [1]
    #Loop through each day. Formula = capital[i+1] = capital[i] (1 - sum(weight[i])) + capital[i] * sum(weights[i]) * price[i+1]/price[i] = capital[i] * (1 + sum(weight[i]) * (price[i+1]/price[i] - 1))
    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i,:]) #Current Capital * proportional # of shares to buy
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i,:])) #Current Capital - Capital Spent on Shares
        net_change = np.dot(shares, np.array(test_data.iloc[i+1,:])) #Value of shares at next day
        capital.append(balance + net_change) #Next days capital = Capital Remaining Today + Profit From Shares
    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1] #Capital from day to day, used to calculate sharpe ratio
    
    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0
    
    with open('sliding_window_output.txt', 'a') as f:
        f.write(f'Bounds: {alloc.lower_bound} {alloc.upper_bound}\n')
        f.write(f'Window Size: {alloc.window_size}\n')
        f.write(f'Method: {alloc.method}\n')
        f.write(f'Sharpe: {sharpe}\n\n')

    return sharpe, capital, weights

sharpe, capital, weights = grading(TRAIN, TEST)
#Sharpe gets printed to command line
print(sharpe)

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Capital")
plt.plot(np.arange(len(TEST)), capital)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Weights")
plt.plot(np.arange(len(TEST)), weights)
plt.legend(TEST.columns)
plt.show()
