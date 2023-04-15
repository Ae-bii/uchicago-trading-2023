#!/usr/bin/env python

from collections import defaultdict
from operator import attrgetter
from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import asyncio
import json
import numpy as np
import math
from math import *
import pandas as pd
from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import norm
from typing import Dict, DefaultDict, Tuple

TICK_SIZE = 0.1
TICKERS_CALL = ["SPY" + str(strike) + "C" for strike in range(65,140,5)]
TICKERS_PUT = ["SPY" + str(strike) + "P" for strike in range(65,140,5)]
TICKERS = TICKERS_CALL + TICKERS_PUT
PARAM_FILE = "clients/params/case2_params.json"



class OptionBot(UTCBot):
    """
    An example bot that reads from a file to set internal parameters during the round
    """
    
    
    """
    Helper Functions
    """
    
    def d1(self,S,K,T,r,sigma):
        return(np.log(S/K)+(r+sigma**2/2.)*T)/(sigma*np.sqrt(T))
    def d2(self,S,K,T,r,sigma):
        return self.d1(S,K,T,r,sigma)-sigma*np.sqrt(T)

    def bs_call(self,S,K,T,r,sigma):
        return S*norm.cdf(self.d1(S,K,T,r,sigma))-K*np.exp(-r*T)*norm.cdf(self.d2(S,K,T,r,sigma))
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

    def delta_put(self,S,K,T,P):
        sigma = self.iv_put(S,K,T,0,P)
        return 100 * (norm.cdf(self.d1(S,K,T,0,sigma)) - 1)

    def gamma_put(self,S,K,T,P):
        sigma = self.iv_put(S,K,T,0,P)
        return 100 * norm.pdf(self.d1(S,K,T,0,sigma))/(S * sigma * np.sqrt(T))

    def vega_put(self,S,K,T,P):
        sigma = self.iv_put(S,K,T,0,P)
        return 100 * norm.pdf(self.d1(S,K,T,0,sigma)) * S * np.sqrt(T)

    def theta_put(self,S,K,T,P):
        sigma = self.iv_put(S,K,T,0,P)
        return 100 * S * norm.pdf(self.d1(S,K,T,0,sigma)) * sigma/(2 * np.sqrt(T))
    
    def update_greek_limits(self):
        for strike in self.option_strikes:
                        
            mid_call = (self._best_bid[f"SPY{strike}C"] + self._best_ask[f"SPY{strike}C"]) / 2
            mid_put = (self._best_bid[f"SPY{strike}P"] + self._best_ask[f"SPY{strike}P"]) / 2
            # if mid_call == 0.0 or mid_put == 0.0:
            #     print(mid_call)
            #     print(mid_put)
            iv_call = vectorized_implied_volatility(mid_call, self.underlying_price[-1], strike, self.time_to_expiry, 0, 'c', q=0, return_as='numpy')[0]
            iv_put = vectorized_implied_volatility(mid_put, self.underlying_price[-1], strike, self.time_to_expiry, 0, 'p', q=0, return_as='numpy')[0]

            
            if not (math.isnan(iv_call) or math.isnan(iv_put)):
                call_option_price = self.compute_options_price('C', self.underlying_price[-1], strike, self.time_to_expiry, iv_call)
                put_option_price = self.compute_options_price('P', self.underlying_price[-1], strike, self.time_to_expiry, iv_put)
                
                # Multiply each of the calulations by our current position of each individual option
                call_position = self.positions.get(f"SPY{strike}C", 0)
                put_position = self.positions.get(f"SPY{strike}P", 0)
                
                self.my_greek_positions["delta"] += call_position*self.delta_call(self.underlying_price[-1], strike, self.time_to_expiry, call_option_price) + put_position*self.delta_put(self.underlying_price[-1], strike, self.time_to_expiry, put_option_price)
                
                self.my_greek_positions["gamma"] += call_position*self.gamma_call(self.underlying_price[-1], strike, self.time_to_expiry, call_option_price) + put_position*self.gamma_put(self.underlying_price[-1], strike, self.time_to_expiry, put_option_price)
                
                self.my_greek_positions["theta"] += call_position*self.theta_call(self.underlying_price[-1], strike, self.time_to_expiry, call_option_price) + put_position*self.theta_put(self.underlying_price[-1], strike, self.time_to_expiry, put_option_price)
                
                self.my_greek_positions["vega"] += call_position*self.vega_call(self.underlying_price[-1], strike, self.time_to_expiry, call_option_price) + put_position*self.vega_put(self.underlying_price[-1], strike, self.time_to_expiry, put_option_price)
    
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
            per_share_val = self.black_scholes('c', underlying_px, strike_px, time_to_expiry, 0.00, volatility)
        elif(flag == 'P' or flag == 'p'):
            per_share_val = self.black_scholes('p', underlying_px, strike_px, time_to_expiry, 0.00, volatility)
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
        self.option_strikes = range(65,140,5)

        self.positions["SPY"] = 0
        for strike in self.option_strikes:
            for flag in ["C", "P"]:
                self.positions[f"SPY{strike}{flag}"] = 0

        self._best_bid: Dict[str, float] = defaultdict(lambda: 0)
        self._second_best_bid: Dict[str, float] = defaultdict(lambda: 0)
        self._best_ask: Dict[str, float] = defaultdict(lambda: 0)
        self._second_best_ask: Dict[str, float] = defaultdict(lambda: 0)
        self.__orders: DefaultDict[str, Tuple[str, float]] = defaultdict(lambda: ("", 0))
        self._fade: DefaultDict[str, float] = defaultdict(lambda: 0)
        self._penny_values: DefaultDict[str, float] = defaultdict(lambda: 0)
        
        self.is_trade = True

        self.underlying_price = [100]
        self.time_tick = 0
        self.time_to_expiry = 1/4 - ((1/4 - 1/12) * (self.time_tick/600))

        self.pnls = [0.0] * 1000
        self.price_path = []
        self.vols = []
        self.greek_limits = {
            "delta": 2000,
            "gamma": 5000,
            "theta": 50000,
            "vega": 1000000
        }
        self.my_greek_positions = {
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
        asyncio.create_task(self.add_trades())
        
    async def add_trades(self):
        while self.time_tick <= 600 and self.is_trade:
            print(self.positions)
            if self.my_greek_positions["delta"] >= 0.8 * self.greek_limits["delta"]:
                for strike in self.option_strikes:
                    mid_call = (self._best_bid[f"SPY{strike}C"] + self._best_ask[f"SPY{strike}C"]) / 2
                    delta_call = self.delta_call(self.underlying_price[-1], strike, self.time_to_expiry, mid_call)
                    if not math.isnan(delta_call) and delta_call != 0:
                        delta_put = delta_call - 1
                        short_call_qty = self.my_greek_positions["delta"] / (delta_call*0.8)
                        short_put_qty = self.my_greek_positions["delta"] / (delta_put*0.8)
                        if int(short_call_qty) > 0:
                            await self.modify_order(f"SPY{strike}C", f"SPY{strike}C", pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK, min(int(short_call_qty), 15))
                            if self.my_greek_positions["delta"] >= self.greek_limits["delta"]:
                                await self.modify_order(f"SPY{strike}C_2", f"SPY{strike}C", pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK, min(int(short_call_qty), 15))

                        if int(short_put_qty) > 0:
                            await self.modify_order(f"SPY{strike}P", f"SPY{strike}P", pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK, min(int(short_put_qty), 15))
                            if self.my_greek_positions["delta"] >= self.greek_limits["delta"]:
                                await self.modify_order(f"SPY{strike}P_2", f"SPY{strike}P", pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK, min(int(short_put_qty), 15))
            elif self.my_greek_positions["delta"] < 0.8 * -self.greek_limits["delta"]:
                for strike in self.option_strikes:
                    mid_call = (self._best_bid[f"SPY{strike}C"] + self._best_ask[f"SPY{strike}C"]) / 2
                    delta_call = self.delta_call(self.underlying_price[-1], strike, self.time_to_expiry, mid_call)
                    if not math.isnan(delta_call) and delta_call != 0:
                        delta_put = delta_call - 1
                        long_call_qty = -self.my_greek_positions["delta"] / (delta_call*0.8)
                        long_put_qty = -self.my_greek_positions["delta"] / (delta_put*0.8)
                        
                        if int(long_call_qty) > 0:
                            await self.modify_order(f"SPY{strike}C", f"SPY{strike}C", pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID, min(int(long_call_qty), 15))
                            if self.my_greek_positions["delta"] <= self.greek_limits["delta"]:
                                await self.modify_order(f"SPY{strike}C_2", f"SPY{strike}C", pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID, min(int(long_call_qty), 15))
                        if int(long_put_qty) > 0:
                            await self.modify_order(f"SPY{strike}P", f"SPY{strike}P", pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID, min(int(long_put_qty), 15))
                            if self.my_greek_positions["delta"] <= -self.greek_limits["delta"]:
                                await self.modify_order(f"SPY{strike}P_2", f"SPY{strike}P", pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID, min(int(long_put_qty), 15))
                        
            if self.my_greek_positions["gamma"] >= 0.8 * self.greek_limits["gamma"]:
                for strike in self.option_strikes:
                    mid_call = (self._best_bid[f"SPY{strike}C"] + self._best_ask[f"SPY{strike}C"]) / 2
                    gamma_call = self.gamma_call(self.underlying_price[-1], strike, self.time_to_expiry, mid_call)
                    if not math.isnan(gamma_call) and gamma_call != 0:
                        gamma_put = gamma_call
                        short_call_qty = self.my_greek_positions["gamma"] / (gamma_call*0.8)
                        short_put_qty = self.my_greek_positions["gamma"] / (gamma_put*0.8)
                        
                        if int(short_call_qty) > 0:
                            await self.modify_order(f"SPY{strike}C", f"SPY{strike}C", pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK, min(int(short_call_qty), 15))
                        if int(short_put_qty) > 0:
                            await self.modify_order(f"SPY{strike}P", f"SPY{strike}P", pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK, min(int(short_put_qty), 15))
            elif self.my_greek_positions["gamma"] < 0.8 * -self.greek_limits["gamma"]:
                for strike in self.option_strikes:
                    mid_call = (self._best_bid[f"SPY{strike}C"] + self._best_ask[f"SPY{strike}C"]) / 2
                    gamma_call = self.gamma_call(self.underlying_price[-1], strike, self.time_to_expiry, mid_call)
                    if not math.isnan(gamma_call) and gamma_call != 0:
                        gamma_put = gamma_call - 1
                        long_call_qty = -self.my_greek_positions["gamma"] / (gamma_call*0.8)
                        long_put_qty = -self.my_greek_positions["gamma"] / (gamma_put*0.8)
                        
                        if int(long_call_qty) > 0:
                            await self.modify_order(f"SPY{strike}C", f"SPY{strike}C", pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID, min(int(long_call_qty), 15))
                        if int(long_put_qty) > 0:
                            await self.modify_order(f"SPY{strike}P", f"SPY{strike}P", pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID, min(int(long_put_qty), 15))
                        
            if self.my_greek_positions["theta"] >= 0.8 * self.greek_limits["theta"]:
                for strike in self.option_strikes:
                    mid_call = (self._best_bid[f"SPY{strike}C"] + self._best_ask[f"SPY{strike}C"]) / 2
                    mid_put = (self._best_bid[f"SPY{strike}P"] + self._best_ask[f"SPY{strike}P"]) / 2
                    theta_call = self.theta_call(self.underlying_price[-1], strike, self.time_to_expiry, mid_call)
                    theta_put = self.theta_put(self.underlying_price[-1], strike, self.time_to_expiry, mid_put)
                    if not (math.isnan(theta_call) or math.isnan(theta_put)) and theta_call != 0 and theta_put != 0:
                        short_call_qty = self.my_greek_positions["theta"] / (theta_call*0.8)
                        short_put_qty = self.my_greek_positions["theta"] / (theta_put*0.8)
                        
                        if int(short_call_qty) > 0:
                            await self.modify_order(f"SPY{strike}C", f"SPY{strike}C", pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK, min(int(short_call_qty),15))
                        if int(short_put_qty) > 0:
                            await self.modify_order(f"SPY{strike}P", f"SPY{strike}P", pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK, min(int(short_put_qty),15))
            elif self.my_greek_positions["theta"] < 0.8 * -self.greek_limits["theta"]:
                for strike in self.option_strikes:
                    mid_call = (self._best_bid[f"SPY{strike}C"] + self._best_ask[f"SPY{strike}C"]) / 2
                    mid_put = (self._best_bid[f"SPY{strike}P"] + self._best_ask[f"SPY{strike}P"]) / 2
                    theta_call = self.theta_call(self.underlying_price[-1], strike, self.time_to_expiry, mid_call)
                    theta_put = self.theta_put(self.underlying_price[-1], strike, self.time_to_expiry, mid_put)
                    if not (math.isnan(theta_call) or math.isnan(theta_put)) and theta_call != 0 and theta_put != 0:
                        long_call_qty = -self.my_greek_positions["theta"] / (theta_call*0.8)
                        long_put_qty = -self.my_greek_positions["theta"] / (theta_put*0.8)
                        
                        if int(long_call_qty) > 0:
                            await self.modify_order(f"SPY{strike}C", f"SPY{strike}C", pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID, min(int(long_call_qty),15))
                        if int(long_put_qty) > 0:
                            await self.modify_order(f"SPY{strike}P", f"SPY{strike}P", pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID, min(int(long_put_qty),15))
                        
            if self.my_greek_positions["vega"] >= 0.8 * self.greek_limits["vega"]:
                for strike in self.option_strikes:
                    mid_call = (self._best_bid[f"SPY{strike}C"] + self._best_ask[f"SPY{strike}C"]) / 2
                    vega_call = self.vega_call(self.underlying_price[-1], strike, self.time_to_expiry, mid_call)
                    if not math.isnan(vega_call) and vega_call != 0:
                        vega_put = vega_call
                        short_call_qty = self.my_greek_positions["vega"] / (vega_call*0.8)
                        short_put_qty = self.my_greek_positions["vega"] / (vega_put*0.8)
                        
                        if int(short_call_qty) > 0:
                            await self.modify_order(f"SPY{strike}C", f"SPY{strike}C", pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK, min(int(short_call_qty),15))
                        if int(short_put_qty) > 0:
                            await self.modify_order(f"SPY{strike}P", f"SPY{strike}P", pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK, min(int(short_put_qty),15))
            elif self.my_greek_positions["vega"] < 0.8 * -self.greek_limits["vega"]:
                for strike in self.option_strikes:
                    mid_call = (self._best_bid[f"SPY{strike}C"] + self._best_ask[f"SPY{strike}C"]) / 2
                    vega_call = self.vega_call(self.underlying_price[-1], strike, self.time_to_expiry, mid_call)
                    if not math.isnan(vega_call) and vega_call != 0:
                        vega_put = vega_call - 1
                        long_call_qty = -self.my_greek_positions["vega"] / (vega_call*0.8)
                        long_put_qty = -self.my_greek_positions["vega"] / (vega_put*0.8)
                        
                        if int(long_call_qty) > 0:
                            await self.modify_order(f"SPY{strike}C", f"SPY{strike}C", pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID, min(int(long_call_qty),15))
                        if int(long_put_qty) > 0:
                            await self.modify_order(f"SPY{strike}P", f"SPY{strike}P", pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID, min(int(long_put_qty),15))
                
            
            for strike in self.option_strikes:
                for flag in ["C", "P"]:
                    best_bid = self._best_bid[f"SPY{strike}{flag}"]
                    best_ask = self._best_ask[f"SPY{strike}{flag}"]
                    
                    mid_price = (best_bid + best_ask) / 2
                    
                    impl_vol = vectorized_implied_volatility(mid_price, self.underlying_price[-1], strike, self.time_to_expiry, 0, flag.lower(), q=0, return_as='numpy')[0]
                    
                    if not math.isnan(impl_vol) and impl_vol != 0:
                        theo = self.compute_options_price(flag, self.underlying_price[-1], strike, self.time_to_expiry, impl_vol)
                        asset = f"SPY{strike}{flag}"
                        
                        penny_bid_price = best_bid + self._penny_values[asset]
                        penny_ask_price = best_ask - self._penny_values[asset]
                        
                        if self.my_greek_positions["theta"] >= 0.8 * self.greek_limits["theta"]:
                            penny_bid_price += self._fade[asset] * self.positions.get(asset, 0)
                            penny_ask_price += self._fade[asset] * self.positions.get(asset, 0)
                        elif self.my_greek_positions["theta"] < 0.8 * -self.greek_limits["theta"]:
                            penny_bid_price -= self._fade[asset] * self.positions.get(asset, 0)
                            penny_ask_price -= self._fade[asset] * self.positions.get(asset, 0)
                        
                        old_bid_id, _ = self.__orders[asset + '_bid']
                        old_ask_id, _ = self.__orders[asset + '_ask']
                        if asset != "SPY65C" and asset != "SPY135P":
                            if self.positions.get(asset, 0) >= -50:
                                if penny_ask_price > theo:
                                    ask_resp = await self.modify_order(f"PENNY_ASK{asset}", asset, pb.OrderSpecType.LIMIT, pb.OrderSpecSide.ASK, 12, round_nearest(penny_ask_price, TICK_SIZE))
                                    
                                    if ask_resp.ok:
                                        self.__orders[asset + '_ask'] = (str(penny_ask_price), ask_resp.order_id)
                            if self.positions.get(asset, 0) <= 50:
                                if penny_bid_price < theo:
                                    bid_resp = await self.modify_order(f"PENNY_BID{asset}", asset, pb.OrderSpecType.LIMIT, pb.OrderSpecSide.BID, 12, round_nearest(penny_bid_price, TICK_SIZE))
                                    
                                    if bid_resp.ok:
                                        self.__orders[asset + '_bid'] = (str(penny_bid_price), bid_resp.order_id)
                                
            await asyncio.sleep(1)
                

    


    async def handle_exchange_update(self, update: pb.FeedMessage):
        self.my_greek_positions = {
            "delta": 0,
            "gamma": 0,
            "theta": 0,
            "vega": 0
        }
        self.time_tick += 1
        kind, _ = betterproto.which_one_of(update, "msg")
        # Competition event messages
        if kind == "pnl_msg":
            # When you hear from the exchange about your PnL, print it out
            print('Realized pnl:', update.pnl_msg.realized_pnl, "| M2M pnl:", update.pnl_msg.m2m_pnl)
            # print(f"Positions: {self.positions}")
        elif kind == "fill_msg":
            # When you hear about a fill you had, update your positions
            fill_msg = update.fill_msg

            if fill_msg.order_side == pb.FillMessageSide.BUY:
                self.positions[fill_msg.asset] += update.fill_msg.filled_qty
            else:
                self.positions[fill_msg.asset] -= update.fill_msg.filled_qty
            # print(update.fill_msg.order_id, "filled:", "BUY" if update.fill_msg.order_side == pb.FillMessageSide.BUY else "SELL", 
            #       update.fill_msg.asset, "@", update.fill_msg.price, "x", update.fill_msg.filled_qty)
            # print(self.positions)
                
        elif kind == "market_snapshot_msg":
            self.books["SPY"] = update.market_snapshot_msg.books["SPY"]
            for asset in TICKERS:
                book = update.market_snapshot_msg.books[asset]
                
                self.books[asset] = update.market_snapshot_msg.books[asset]
                best_bid = max(book.bids, key=attrgetter('px'), default=None)
                second_best_bid = book.bids[1] if len(book.bids) > 1 else None
                
                best_ask = min(book.asks, key=attrgetter('px'), default=None)
                second_best_ask = book.asks[1] if len(book.asks) > 1 else None
                
                self._best_bid[asset] = float(best_bid.px) if best_bid is not None else 0
                self._second_best_bid[asset] = float(second_best_bid.px) if second_best_bid is not None else 0
                
                self._best_ask[asset] = float(best_ask.px) if best_ask is not None else 0
                self._second_best_ask[asset] = float(second_best_ask.px) if second_best_ask is not None else 0
                
                
            book = update.market_snapshot_msg.books["SPY"]            
            if (len(book.bids) > 0) and (len(book.asks) > 0):
                self.underlying_price.append((
                    float(book.bids[0].px) + float(book.asks[0].px)
                ) / 2)
                self.update_greek_limits()
                print(self.my_greek_positions)
                
            # if (self.time_tick < 599):
            #     await self.update_options_quotes()
            # print(self.positions)
            
        elif (
            kind == "generic_msg"
            and update.generic_msg.event_type == pb.GenericMessageType.MESSAGE
        ):
            resp = await self.get_positions()
            if resp.ok:
                self.positions = resp.positions
            # The platform will regularly send out what day it currently is (starting from day 0 at
            # the start of the case) 
            # print(f"Positions: {self.positions}")
            # self.price_path.append(self.underlying_price)
            # print(f"New Price: {self.underlying_price}")
            
            # print("Underlying ", self.underlying_price)
            # if (self.time_tick == 599):
            #     # self.market_closed()
            #     pass
                

    async def handle_read_params(self):
        while True:
            try:
                self.params = json.load(open(PARAM_FILE, "r"))
                for asset in TICKERS:
                    # self._spread[asset] = params[asset_str]['edge']
                    self.is_trade = True if self.params['is_trading'] == 1 else False
                    self._penny_values[asset] = self.params['penny_values'][asset + '_Penny']
                    self._fade[asset] = self.params['fade'][asset]
                    # self._quantity[asset] = params[asset_str]['size']
                    # self._slack[asset] = params[asset_str]['slack']
            except:
                print("Unable to read file " + PARAM_FILE)

            await asyncio.sleep(1)

def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))

if __name__ == "__main__":
    start_bot(OptionBot)
