#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aditya


Monte Carlo delta-hedging simulator for European options.

Simulates underlying price paths using Geometric Brownian Motion,
dynamically rebalances a Black-Scholes delta hedge, accrues the cash
account, and calculates terminal hedging P&L.

Key inputs:
- implied volatility: used for option valuation and hedge ratios
- realised volatility: used to simulate underlying price paths
- number_of_simulations: number of Monte Carlo paths

Assumptions:
- European-style options
- No dividends
- Constant risk-free rate
- Constant implied and realised volatility
- Discrete daily hedging
    
"""

import numpy as np
import scipy.stats as stats

class Deltahedging: 
    def __init__(self,spot,strike,risk_free_rate,Time_to_Maturity,sigma_implied,sigma_real,option_type,number_of_simulations):
        self.s = spot
        self.k = strike
        self.r= risk_free_rate
        self.maturity = Time_to_Maturity
        self.sigma_imp = sigma_implied
        self.sigma_real =sigma_real
        self.option_type = option_type
        self.steps = int(252 * self.maturity)
        self.simulation = number_of_simulations
        self.dt = 1/252
        
       
        if self.s <= 0:
            raise ValueError("spot must be positive")
    
        if self.k <= 0:
            raise ValueError("strike must be positive")
    
        if self.maturity <= 0:
            raise ValueError("Time_to_Maturity must be positive")
    
        if self.sigma_imp <= 0:
            raise ValueError("sigma_implied must be positive")
    
        if self.sigma_real < 0:
            raise ValueError("sigma_real cannot be negative")
    
        if self.option_type not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'")
    
        if self.simulation <= 0:
            raise ValueError("number_of_simulations must be positive")
      
        
        
    
    def compute_distribution(self,updated_spot,maturity):
        d1 = ((np.log(updated_spot/self.k)+((self.r+(0.5*self.sigma_imp**2)))*maturity)) / (self.sigma_imp*np.sqrt(maturity))
        d2 = d1 - (self.sigma_imp*np.sqrt(maturity))
        
        return d1,d2
    
    
    def bs_price(self,updated_spot,maturity):
       
        d1,d2 =(self.compute_distribution(updated_spot, maturity))
        
        Nd1 = stats.norm.cdf(d1)
        Nd2 = stats.norm.cdf(d2)
        Nd1_put = stats.norm.cdf(-d1)
        Nd2_put = stats.norm.cdf(-d2)
        if self.option_type == 'call':
            option_price = (updated_spot*Nd1)-(Nd2*(self.k*np.exp(-self.r*maturity)))
        elif self.option_type == 'put':
            option_price = Nd2_put*(self.k*np.exp(-self.r*maturity))-(updated_spot*Nd1_put)

        return option_price
    
    def delta(self,updated_spot,maturity):
      
        d1,d2 =(self.compute_distribution(updated_spot, maturity))
        
        if self.option_type == 'call':
            delta = stats.norm.cdf(d1)
        elif self.option_type == 'put':
            delta = -stats.norm.cdf(-d1)
        return delta
    
    def gamma(self,updated_spot,maturity):
        
        d1,d2 =(self.compute_distribution(updated_spot, maturity))
        
        nd1 = stats.norm.pdf(d1)
        gamma = nd1/(updated_spot*self.sigma_imp*np.sqrt(maturity))
        return gamma
    
    def new_price(self,old_price):
        drift = (self.r-0.5* self.sigma_real**2)*self.dt
        randomness = np.random.normal(0,1)*np.sqrt(self.dt)*self.sigma_real
        next_price = old_price * np.exp((drift + randomness))
        return next_price
    
    def hedge_position(self): 
       
        final_payoff=[]
        stock_value_expiry =[]
        final_cash=[]
        terminal_pnls = []
        
        for s in range(self.simulation):
            cash_position=[]
            option_prices=[]
            current_stock_position=[]
            delta_timeline=[]
            spot_prices =[self.s]
          
            
            for t in range(self.steps):
                
                ttm = self.maturity - (t*self.dt)
                spot_price = spot_prices[t]
                option_price_current = self.bs_price(spot_price,ttm)
                option_prices.append(option_price_current)
                delta_t = self.delta(spot_price,ttm)
                delta_timeline.append(delta_t)
                    
                if t == 0:
                    trade = -delta_t
                    current_cash = -option_price_current -(trade*spot_price)
                    cash_position.append(current_cash)
                    new_position = trade
                    current_stock_position.append(new_position)
                      
                else:
                    trade = -delta_t - current_stock_position[t-1]
                    current_cash = (cash_position[t-1]*np.exp(self.r*self.dt)) - (trade*spot_price)
                    new_position = current_stock_position[t-1] + trade
                    current_stock_position.append(new_position)
                    cash_position.append(current_cash)
                    
                next_price = self.new_price(spot_price)
                spot_prices.append(next_price)
                        
            portfolio_value = np.array(option_prices)+(np.array(spot_prices[:self.steps])* np.array(current_stock_position))+np.array(cash_position)
        
            if self.option_type == 'call' :
                payoff = max(spot_prices[-1]-self.k,0)
                final_payoff.append(payoff)
            else:
                payoff = max(self.k-spot_prices[-1],0)
                final_payoff.append(payoff)
            
            value_expiry = current_stock_position[-1] * spot_prices[-1]
            stock_value_expiry.append(value_expiry)
            latest_cash = cash_position[-1] * np.exp(self.r * self.dt)
            final_cash.append(latest_cash)

            terminal_pnl = payoff+ value_expiry+latest_cash
            terminal_pnls.append(terminal_pnl)
            
            mean_pnl = np.mean(terminal_pnls)
            std_pnl = np.std(terminal_pnls)
            
        return mean_pnl, std_pnl
       
 

if __name__ == "__main__":
       
    sigma_real = [0.10, 0.20, 0.30, 0.40]
    for s in sigma_real:
        
        np.random.seed(99)
        h1 = Deltahedging(100, 100, 0.05,1,0.20,s,'put', 1000)  
        mean_pnl, std_pnl = h1.hedge_position()              
    
        print(f"Mean terminal hedged P&L: {mean_pnl} for {s} as realised vol" )
        print(f"Standard deviation of P&L: {std_pnl} for {s} as realised vol ")
                
            
            
            
        
        
        
        
        
        
        
        