
"""
    Blackscholes pricer for European call and put options.

    Calculates option value and analytical Greeks including
    delta, gamma, vega, volga, rho and theta.
    Inputs:
        - spot: underlying spot price
        - strike: option strike price
        - risk_free_rate: annualised continuously compounded rate
        - time_to_maturity: trading days to expiry (252 days = 1 year)
        - volatility: annualised volatility
    Model assumptions:
    - European-style options
    - No dividends
    - Constant volatility
    - Constant risk-free interest rate
    - Lognormal underlying price dynamics
"""



import numpy as np
import scipy.stats as stats

class Blackscholespricer:
    def __init__(self,spot,strike,risk_free_rate,time_to_maturity,volatility):
        self.s =spot
        self.k = strike
        self.t = time_to_maturity/252
        self.sigma = volatility
        self.r = risk_free_rate
       
        if self.s <= 0:
            raise ValueError("spot must be positive")   

        if self.k <= 0:
            raise ValueError("strike must be positive")

        if self.t <= 0:
            raise ValueError("time_to_maturity must be positive")

        if self.sigma <= 0:
            raise ValueError("volatility must be positive")
        
        self.d1 , self.d2 = self.compute_distribution()
        
    def compute_distribution(self):
        d1 = ((np.log(self.s/self.k)+((self.r+(0.5*self.sigma**2)))*self.t)) / (self.sigma*np.sqrt(self.t))
        d2 = d1 - (self.sigma*np.sqrt(self.t))
        
        return d1,d2
    
    def bs_price(self,option_type):
        Nd1 = stats.norm.cdf(self.d1)
        Nd2 = stats.norm.cdf(self.d2)
        Nd1_put = stats.norm.cdf(-self.d1)
        Nd2_put = stats.norm.cdf(-self.d2)
        if option_type == 'call':
            option_price = (self.s*Nd1)-(Nd2*(self.k*np.exp(-self.r*self.t)))
        elif option_type == 'put':
            option_price = Nd2_put*(self.k*np.exp(-self.r*self.t))-(self.s*Nd1_put)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return option_price
    
    def delta(self,option_type):
      
        if option_type == 'call':
            delta = stats.norm.cdf(self.d1)
        elif option_type == 'put':
            delta = -stats.norm.cdf(-self.d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return delta
    
    def gamma(self):
      
        nd1 = stats.norm.pdf(self.d1)
        gamma = nd1/(self.s*self.sigma*np.sqrt(self.t))
        return gamma
    
    def vega(self):
        
        nd1 = stats.norm.pdf(self.d1)
        vega = self.s*np.sqrt(self.t)*nd1
        volga = vega * (self.d1*self.d2/(self.sigma))
        return vega,volga

    def rho(self,option_type):
        
        Nd2 = stats.norm.cdf(self.d2)
        Nd2_put = stats.norm.cdf(-self.d2)
        if option_type == 'call':
            rho = Nd2*np.exp(-self.r*self.t)*self.k*self.t
        elif option_type == 'put':
            rho = -Nd2_put*np.exp(-self.r*self.t)*self.k*self.t
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return rho
    
    def theta(self,option_type):
        nd1 = stats.norm.pdf(self.d1)
        Nd2 = stats.norm.cdf(self.d2)
        Nd2_put = stats.norm.cdf(-self.d2)
        if option_type == 'call':
            theta = ((-self.s*self.sigma*nd1)/(2*np.sqrt(self.t)))-(self.r*self.k*np.exp(-self.r*self.t)*Nd2)
        elif option_type == 'put':
            theta = ((-self.s*self.sigma*nd1)/(2*np.sqrt(self.t)))+(self.r*self.k*np.exp(-self.r*self.t)*Nd2_put)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        return theta

if __name__ == "__main__":     
    
    o1 = Blackscholespricer(100, 100,0.05,180 ,0.20)
    call_price = o1.bs_price('call')
    put_price = o1.bs_price('put')
    
    print(f"Call price: {call_price}")
    print(f"Put price: {put_price}")
    
    parity_difference = ( call_price - put_price - (o1.s - o1.k * np.exp(-o1.r * o1.t)) )

    print(f"Put call parity difference: {parity_difference}")
    print(f" Option price:{o1.bs_price('call')}")
    print(f" Delta: {o1.delta('call')}")
    print(f"Gamma: {o1.gamma()}")
    print(f"Vega: {o1.vega()[0]}")
    print(f"Volga: {o1.vega()[1]}")
    print(f"Rho : {o1.rho('call')}")
    print(f"Theta: {o1.theta('call')}")