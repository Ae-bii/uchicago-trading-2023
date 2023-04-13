#!/usr/bin/env python

from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import asyncio
import orjson
import numpy as np
from math import *
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import norm

PARAM_FILE = "./params/case2_params.json"



class OptionBot(UTCBot):
    """
    An example bot that reads from a file to set internal parameters during the round
    """
    def __init__(self):
        self.option_strikes = range(65,140,5)
    
    
    """
    Helper Functions
    """
    
    def d1(self,S,K,T,r,sigma):
        return(np.log(S/K)+(r+sigma**2/2.)*T)/(sigma*np.sqrt(T))
    def d2(self,S,K,T,r,sigma):
        return self.d1(S,K,T,r,sigma)-sigma*np.sqrt(T)

    def bs_call(self,S,K,T,r,sigma):
        return S*norm.cdf(self.d1(S,K,T,r,sigma))-K*np.exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))
    def bs_put(self,S,K,T,r,sigma):
        return K*np.exp(-r*T)-S+self.bs_call(S,K,T,r,sigma)

    # Implied Volatility:
    def iv_call(self,S,K,T,r,C):
        return max(0, fsolve((lambda sigma: np.abs(self.bs_call(S,K,T,r,sigma) - C)), [1])[0])
                        
    def iv_put(self,S,K,T,r,P):
        return max(0, fsolve((lambda sigma: np.abs(self.bs_put(S,K,T,r,sigma) - P)), [1])[0])

    def delta_call(self,S,K,T,C):
        sigma = self.iv_call(S,K,T,0,C)
        return 100 * norm.cdf(self.d1(S,K,T,0,sigma))

    def gamma_call(self,S,K,T,C):
        sigma = self.iv_call(S,K,T,0,C)
        return 100 * norm.pdf(self.d1(S,K,T,0,sigma))/(S * sigma * np.sqrt(T))

    def vega_call(self,S,K,T,C):
        sigma = self.iv_call(S,K,T,0,C)
        return 100 * norm.pdf(self.d1(S,K,T,0,sigma)) * S * np.sqrt(T)

    def theta_call(self,S,K,T,C):
        sigma = self.iv_call(S,K,T,0,C)
        return 100 * S * norm.pdf(self.d1(S,K,T,0,sigma)) * sigma/(2 * np.sqrt(T))

    def delta_put(self,S,K,T,C):
        sigma = self.iv_put(S,K,T,0,C)
        return 100 * (norm.cdf(self.d1(S,K,T,0,sigma)) - 1)

    def gamma_put(self,S,K,T,C):
        sigma = self.iv_put(S,K,T,0,C)
        return 100 * norm.pdf(self.d1(S,K,T,0,sigma))/(S * sigma * np.sqrt(T))

    def vega_put(self,S,K,T,C):
        sigma = self.iv_put(S,K,T,0,C)
        return 100 * norm.pdf(self.d1(S,K,T,0,sigma)) * S * np.sqrt(T)

    def theta_put(self,S,K,T,C):
        sigma = self.iv_put(S,K,T,0,C)
        return 100 * S * norm.pdf(self.d1(S,K,T,0,sigma)) * sigma/(2 * np.sqrt(T))
    # Cumulative standard normal distribution
    def cdf(self,x):
        return (1.0 + erf(x / sqrt(2.0))) / 2.0

    # Call Price based on Black Scholes Model
    # Parameters
    #   underlying_price: Price of underlying asset
    #   exercise_price: Exercise price of the option
    #   time_in_years: Time to expiration in years (ie. 33 days to expiration is 33/365)
    #   risk_free_rate: Risk free rate (ie. 2% is 0.02)
    #   volatility: Volatility percentage (ie. 30% volatility is 0.30)
    #             per_share_val = bs.black_scholes('c', underlying_px, strike_px, time_to_expiry, 0.00, volatility)
    def black_scholes(self,flag, underlying_price, exercise_price, time_in_years, risk_free_rate, volatility):
        if flag == 'c' or flag == 'C':
            d1 = (log(underlying_price / exercise_price) + risk_free_rate * time_in_years) / (volatility * sqrt(time_in_years)) + 0.5 * volatility * sqrt(time_in_years)
            d2 = d1 - (volatility * sqrt(time_in_years))
            
            return underlying_price * self.cdf(d1) - exercise_price * exp(-time_in_years * risk_free_rate) * self.cdf(d2)
        else:
            return self.black_scholes_put(underlying_price, exercise_price, time_in_years, risk_free_rate, volatility)
    # Put Price based on Black Scholes Model
    # Parameters
    #   underlying_price: Price of underlying asset
    #   exercise_price: Exercise price of the option
    #   time_in_years: Time to expiration in years (ie. 33 days to expiration is 33/365)
    #   risk_free_rate: Risk free rate (ie. 2% is 0.02)
    #   volatility: Volatility percentage (ie. 30% volatility is 0.30)
    def black_scholes_put(self,underlying_price, exercise_price, time_in_years, risk_free_rate, volatility):
        return self.black_scholes('c',underlying_price, exercise_price, time_in_years, risk_free_rate, volatility) + exercise_price * exp(-risk_free_rate * time_in_years) - underlying_price
    
    def compute_vol_estimate(self) -> float:
        """
        This function is used to provide an estimate of underlying's volatility. Because this is
        an example bot, we just use a placeholder value here. We recommend that you look into
        different ways of finding what the true volatility of the underlying is.
        """

        if(len(self.price_path) <= 20):
            return 0.2
        else:
            stdev = np.std(self.price_path[-100:])
            volatility = 0.9* np.log(stdev/2.5 + 0.375) + 0.9
            return volatility
    
    def compute_options_price(
        self,   
        flag: str,
        underlying_px: float,
        strike_px: float,
        time_to_expiry: float,
        volatility: float
    ) -> float:
        """
        This function should compute the price of an option given the provided parameters. Some
        important questions you may want to think about are:
            - What are the units associated with each of these quantities?
            - What formula should you use to compute the price of the option?
            - Are there tricks you can use to do this more quickly?
        You may want to look into the py_vollib library, which is installed by default in your
        virtual environment.
        """
        per_share_val = 0
        if(flag == 'C' or flag == 'c'):
            per_share_val = self.my_bs('c', underlying_px, strike_px, time_to_expiry, 0.00, volatility)
        elif(flag == 'P' or flag == 'p'):
            per_share_val = self.my_bs('p', underlying_px, strike_px, time_to_expiry, 0.00, volatility)
        if (per_share_val < 0.1):
            per_share_val = 0.1
        return np.round(per_share_val, 1)
    
    
    """
    Trading Logic
    """
    
    async def handle_round_started(self):
        # This variable will be a map from asset names to positions. We start out by initializing it
        # to zero for every asset.
        self.positions = {}

        self.positions["UC"] = 0
        for strike in self.option_strikes:
            for flag in ["C", "P"]:
                self.positions[f"UC{strike}{flag}"] = 0

        # Stores the current day (starting from 0 and ending at 5). This is a floating point number,
        # meaning that it includes information about partial days
        self.current_day = 0

        # Stores the current value of the underlying asset
        self.underlying_price = 100
        self.time_tick = 0
        self.pnls = [0.0] * 1000
        self.price_path = []
        self.puts100 = []
        self.calls100 = []
        self.vols = []
        self.C100_price = 0
        self.greek_limits = {
            "delta": 2000,
            "gamma": 5000,
            "theta": 50000,
            "vega": 1000000
        }
        self.my_greek_limits = {
            "delta": 0,
            "gamma": 0,
            "theta": 0,
            "vega": 0
        }
        self.books={}
        self.safe_buy = 0
        self.max_contracts_left = 0
        await asyncio.sleep(0.1)
        asyncio.create_task(self.handle_read_params())
    
    def add_trades(self):
        requests = []
        day = np.floor(self.current_day)
        dte = 26-day
        time_to_expiry = dte / 252
        theo = self.compute_options_price('p', self.underlying_price, 100, time_to_expiry, self.compute_vol_estimate())
        
        if(len(self.price_path) == 200):
            for i in range(20):
                label = f"covid_{i}"
                requests.append(
                    self.modify_order(
                        label,
                        "UC100P",
                        pb.OrderSpecType.LIMIT,
                        pb.OrderSpecSide.BID,
                        1,
                        theo,
                    )
                )
        if(theo*100 > 900):
            for i in range(20):
                label = f"covid_{i}"
                requests.append(
                    self.modify_order(
                        label,
                        "UC100P",
                        pb.OrderSpecType.LIMIT,
                        pb.OrderSpecSide.ASK,
                        1,
                        theo,
                    )
                )
        return requests

    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")
        # Competition event messages
        if kind == "generic_msg":
            msg = update.generic_msg.message
            print(msg)

    async def handle_read_params(self):
        while True:
            try:
                self.params = orjson.loads(open(PARAM_FILE, "r").read())
            except:
                print("Unable to read file " + PARAM_FILE)

            await asyncio.sleep(1)


if __name__ == "__main__":
    start_bot(OptionBot)
