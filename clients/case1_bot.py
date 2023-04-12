#!/usr/bin/env python

from collections import defaultdict
from typing import DefaultDict, Dict, Tuple
from utc_bot import UTCBot, start_bot
import math
import numpy as np
from proto import utc_bot as pb
import betterproto
import asyncio
import re
import orjson

DAYS_IN_MONTH = 21
DAYS_IN_YEAR = 252
INTEREST_RATE = 0.02
NUM_FUTURES = 14
TICK_SIZE = 0.01
FUTURE_CODES = [chr(ord('A') + i) for i in range(NUM_FUTURES)] # Suffix of monthly future code
CONTRACTS = ['SBL'] +  ['LBS' + c for c in FUTURE_CODES] + ['LLL']

WEATHER_LAG_CORRELATIONS = [0.8273928658695076,
                            0.8490646085184801,
                            0.8660726823774236,
                            0.8819633686272389,
                            0.8938345173210757,
                            0.9028685720889777,
                            0.9114565561827758,
                            0.9161042572964535,
                            0.9194070826557654,
                            0.9180806552651403,
                            0.917776546901705,
                            0.9138785469586278]
SBL_MEAN = -6.046839389007168e-05
SBL_VAR = 9.113656672214566e-05


PARAM_FILE = "./params/case1_params.json"

class Case1Bot(UTCBot):
    etf_suffix = ''

    soy_prices = np.zeros(DAYS_IN_YEAR)

    async def create_etf(self, qty: int):
        '''
        Creates qty amount the ETF basket
        DO NOT CHANGE
        '''
        if len(self.etf_suffix) == 0:
            return pb.SwapResponse(False, "Unsure of swap")
        return await self.swap("create_etf_" + self.etf_suffix, qty)

    async def redeem_etf(self, qty: int):
        '''
        Redeems qty amount the ETF basket
        DO NOT CHANGE
        '''
        if len(self.etf_suffix) == 0:
            return pb.SwapResponse(False, "Unsure of swap")
        return await self.swap("redeem_etf_" + self.etf_suffix, qty) 
    
    async def days_to_expiry(self, asset: str):
        '''
        Calculates days to expiry for the future
        '''
        future = ord(asset[-1]) - ord('A')
        expiry = 21 * (future + 1)
        return expiry - self._day

    async def handle_exchange_update(self, update: pb.FeedMessage):
        '''
        Handles exchange updates
        '''
        kind, _ = betterproto.which_one_of(update, "msg")
        #Competition event messages
        if kind == "generic_msg":
            msg = update.generic_msg.message
            
            # Used for API DO NOT TOUCH
            if 'trade_etf' in msg:
                self.etf_suffix = msg.split(' ')[1]
                
            # Updates current weather
            if "Weather" in update.generic_msg.message:
                msg = update.generic_msg.message
                weather = float(re.findall("\d+\.\d+", msg)[0])
                self._weather_log.append(weather)
                
            # Updates date
            if "Day" in update.generic_msg.message:
                self._day = int(re.findall("\d+", msg)[0])
                            
            # Updates positions if unknown message (probably etf swap)
            else:
                resp = await self.get_positions()
                if resp.ok:
                    self.positions = resp.positions
                    
        elif kind == "MarketSnapshotMessage":
            for asset in CONTRACTS:
                book = update.market_snapshot_msg.books[asset]
                self._best_bid[asset] = float(book.bids[0].px)
                self._best_ask[asset] = float(book.asks[0].px)

    async def handle_round_started(self):
        ### Current day
        self._day = 0
        ### Best Bid in the order book
        self._best_bid: Dict[str, float] = defaultdict(
            lambda: 0
        )
        ### Best Ask in the order book
        self._best_ask: Dict[str, float] = defaultdict(
            lambda: 0
        )
        ### Order book for market making
        self.__orders: DefaultDict[str, Tuple[str, float]] = defaultdict(
            lambda: ("", 0)
        )
        ### TODO Recording fair price for each asset
        self._fair_price: DefaultDict[str, float] = defaultdict(
            lambda: ("", 0)
        )
        ### TODO spread fair price for each asset
        self._spread: DefaultDict[str, float] = defaultdict(
            lambda: ("", 0)
        )

        ### TODO order size for market making positions
        self._quantity: DefaultDict[str, int] = defaultdict(
            lambda: ("", 0)
        )
        
        ### List of weather reports
        self._weather_log = []
        
        await asyncio.sleep(.1)
        ###
        ### TODO START ASYNC FUNCTIONS HERE
        ###
        # asyncio.create_task(self.example_redeem_etf())
        asyncio.create_task(self.handle_read_params())
        
        # Starts market making for each asset
        for asset in CONTRACTS:
            asyncio.create_task(self.make_market_asset(asset))

    # This is an example of creating and redeeming etfs
    # You can remove this in your actual bots.
    async def example_redeem_etf(self):
        while True:
            redeem_resp = await self.redeem_etf(1)
            create_resp = await self.create_etf(5)
            await asyncio.sleep(1)

    ### Helpful ideas
    async def calculate_risk_exposure(self):
        pass
    
    async def calculate_asset_price(self):
        if self._day == 0:
            pred = self._best_ask['SBL'] + self._best_bid['SBL']
            conf = -1
        else:
            corr = np.mean(WEATHER_LAG_CORRELATIONS[min(len(self._weather_log) - 1, 8)])
            ind = np.mean(self._weather_log[min(len(self._weather_log) - 1, 8)])
            pred = self.soy_prices[self._day - 1] * np.exp((SBL_MEAN - SBL_VAR / 2) \
                        + np.sqrt(SBL_VAR) * (corr * ind \
                        + np.sqrt(1 - corr ** 2) * np.random.normal(0, 1)))
            # 1.96 = +/-1.96 std deviations, which represents 95% confidence
            conf = 1.96 * pred * np.sqrt(SBL_VAR) * ((1 - corr ** 2) + corr * ind)
        return pred, conf

    async def calculate_future_price(self, asset: str):
        underlying_asset_price = await self.calculate_asset_price()
        days = await self.days_to_expiry(asset)
        return underlying_asset_price * (1 + INTEREST_RATE * days / DAYS_IN_YEAR)
 
    async def make_market_asset(self, asset: str):
        while self._day <= DAYS_IN_YEAR:
            ## Old prices
            ub_oid, ub_price = self.__orders["underlying_bid_{}".format(asset)]
            ua_oid, ua_price = self.__orders["underlying_ask_{}".format(asset)]
            
            bid_px = self._fair_price[asset] - self._spread[asset]
            ask_px = self._fair_price[asset] + self._spread[asset]
            
            # If the underlying price moved first, adjust the ask first to avoid self-trades
            if (bid_px + ask_px) > (ua_price + ub_price):
                order = ["ask", "bid"]
            else:
                order = ["bid", "ask"]

            for d in order:
                if d == "bid":
                    order_id = ub_oid
                    order_side = pb.OrderSpecSide.BID
                    order_px = bid_px
                else:
                    order_id = ua_oid
                    order_side = pb.OrderSpecSide.ASK
                    order_px = ask_px

                r = await self.modify_order(
                        order_id = order_id,
                        asset_code = asset,
                        order_type = pb.OrderSpecType.LIMIT,
                        order_side = order_side,
                        qty = self._quantity[asset],
                        px = round_nearest(order_px, TICK_SIZE), 
                    )

                self.__orders[f"underlying_{d}_{asset}"] = (r.order_id, order_px)

    async def handle_read_params(self):
        while True:
            try:
                self.params = orjson.loads(open(PARAM_FILE, "r").read())
            except:
                print("Unable to read file " + PARAM_FILE)

            await asyncio.sleep(1)

def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))             

if __name__ == "__main__":
    start_bot(Case1Bot)
