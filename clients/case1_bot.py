#!/usr/bin/env python

from collections import defaultdict
from typing import DefaultDict, Dict, Tuple, List
from utc_bot import UTCBot, start_bot
import math
import numpy as np
from proto import utc_bot as pb
import betterproto
import asyncio
import re
import json
from operator import attrgetter

DAYS_IN_MONTH = 21
DAYS_IN_YEAR = 252
INTEREST_RATE = 0.02
NUM_FUTURES = 14
TICK_SIZE = 0.01
FUTURE_CODES = [chr(ord('A') + i) for i in range(NUM_FUTURES)] # Suffix of monthly future code
CONTRACTS = ['SBL'] +  ['LBS' + c for c in FUTURE_CODES] + ['LLL']

WEATHER_LAG_CORRELATIONS = [0.7992952623820577,
                            0.8273928658695076,
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

PARAM_FILE = "clients/params/case1_params.json"

class Case1Bot(UTCBot):
    etf_suffix = ''

    soy_prices = np.zeros(DAYS_IN_YEAR)
    soy_prices[0] = 53.803717019730705

    size_base = 25
    level_sizes = [10, 5, 5]
    level_spreads = [0.05, 0.15, 0.25]

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
    
        elif kind == "market_snapshot_msg":
            for asset in CONTRACTS:
                book = update.market_snapshot_msg.books[asset]
                best_bid = max(book.bids, key=attrgetter('px'), default=None)
                best_ask = min(book.asks, key=attrgetter('px'), default=None)
                self._best_bid[asset] = (float(best_bid.px), best_bid.qty) if best_bid is not None else (0, 0)
                self._best_ask[asset] = (float(best_ask.px), best_ask.qty) if best_ask is not None else (0, 0)
            
        elif kind == "pnl_msg":
            print('Realized pnl:', update.pnl_msg.realized_pnl, "| M2M pnl:", update.pnl_msg.m2m_pnl)

        elif kind == "fill_msg":
            # print(update.fill_msg.order_id, "filled:", "BUY" if update.fill_msg.order_side == "BUY" else "SELL", 
            #       update.fill_msg.asset, "@", update.fill_msg.price, "x", update.fill_msg.filled_qty)
            order = self.__id_to_order[update.fill_msg.order_id]
            prev_order = self.__orders[order]
            self.__orders[order] = (prev_order[0], prev_order[1], prev_order[2]+ \
                                        (1 if update.fill_msg.order_side == "BUY" else -1) * update.fill_msg.filled_qty)

        elif kind == "order_cancelled_msg":
            ids = update.order_cancelled_msg.order_ids
            # print(ids, "cancelled:", update.order_cancelled_msg.asset, 
            #       "intentional!" if update.order_cancelled_msg.intentional else "not intentional!")
            for id in ids:
                order = self.__id_to_order[id]
                prev_order = self.__orders[order]
                self.__orders[order] = (prev_order[0], prev_order[1], 0)

    async def handle_round_started(self):
        ### Current day
        self._day = 0
        ### Best Bid in the order book
        self._best_bid: Dict[str, Tuple[float, int]] = defaultdict(lambda: (0, 0))
        ### Best Ask in the order book
        self._best_ask: Dict[str, Tuple[float, int]] = defaultdict(lambda: (0, 0))
        ### Order book for market making
        self.__orders: DefaultDict[str, Tuple[str, float, int]] = defaultdict(lambda: ("", 0, 0))
        self.__id_to_order: DefaultDict[str, str] = defaultdict(lambda: "")
        ### TODO Recording fair price for each asset
        self._fair_price: DefaultDict[str, float] = defaultdict(lambda: 0)
        ### TODO spread fair price for each asset
        # self._spread: DefaultDict[str, float] = defaultdict(lambda: 0)
        self._fade: DefaultDict[str, float] = defaultdict(lambda: 0)
        # self._slack: DefaultDict[str, float] = defaultdict(lambda: 0)
        ### TODO order size for market making positions
        self._quantity: DefaultDict[str, int] = defaultdict(lambda: 0)
        
        ### List of weather reports
        self._weather_log = []
        
        await asyncio.sleep(.1)
        """
        STARTING ASYNC FUNCTIONS
        """
        asyncio.create_task(self.print_positions())
        asyncio.create_task(self.handle_read_params())
        asyncio.create_task(self.arbitrage_etf())

        # Starts market making for each asset
        for asset in CONTRACTS:
            if asset == 'LLL':
                continue
            asyncio.create_task(self.make_market_asset(asset))

    async def calculate_asset_price(self, asset: str):
        if asset == 'SBL':
            await self.calculate_soy_price()
        elif asset == 'LLL':
            await self.calculate_etf_price()
        else:
            await self.calculate_future_price(asset)

    async def calculate_soy_price(self):
        # corr = WEATHER_LAG_CORRELATIONS[min(len(self._weather_log) - 1, 9)]
        # ind = self._weather_log[-min(len(self._weather_log), 9)]
        corr = 0.9
        ind = 0.14
        pred = self.soy_prices[self._day] * np.exp((SBL_MEAN - SBL_VAR / 2) \
                    + np.sqrt(SBL_VAR) * (corr * ind \
                    + np.sqrt(1 - corr ** 2) * np.random.normal(0, 1)))
        # 1.96 = +/-1.96 std deviations, which represents 95% confidence
        conf = 1.96 * pred * np.sqrt(SBL_VAR) * ((1 - corr ** 2) + corr * ind)

        self._fair_price['SBL'] = pred
        # self._spread['SBL'] = conf

    async def calculate_future_price(self, asset: str):
        if self._fair_price['SBL'] == 0:
            await self.calculate_soy_price()
        days = await self.days_to_expiry(asset)
        self._fair_price[asset] = self._fair_price['SBL'] * (1 + INTEREST_RATE * days / DAYS_IN_YEAR)

    async def calculate_etf_price(self):
        mth = ord('A') + round(self._day / 22)
        total_p = 0
        for i in range(mth, mth + 3):
            ticker = 'LBS' + chr(i)
            if self._fair_price[ticker] == 0:
                await self.calculate_future_price(ticker)
            total_p += self._fair_price[ticker]
        self._fair_price['LLL'] = total_p
 
    async def make_market_asset(self, asset: str):
        while self._day <= DAYS_IN_YEAR:
            # Checking that we aren't doing penny-in on our own orders!
            if self._best_bid[asset][0] == self.__orders[asset + '_bid'][1]:
                penny_bid_price = self.__orders[asset + '_bid'][1]
            else:
                penny_bid_price = self._best_bid[asset][0] + 0.01 - self._fade[asset] * self.positions.get(asset, 0) / 100

            if self._best_ask[asset][0] == self.__orders[asset + '_ask'][1]:
                penny_ask_price = self.__orders[asset + '_ask'][1]
            else:
                penny_ask_price = self._best_ask[asset][0] - 0.01 - self._fade[asset] * self.positions.get(asset, 0) / 100

            if penny_ask_price - penny_bid_price > 0:
                old_bid_id, _, __ = self.__orders[asset + '_bid']
                old_ask_id, _, __ = self.__orders[asset + '_ask']

                bid_resp = await self.modify_order(
                    old_bid_id,
                    asset,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.BID,
                    self.size_base,
                    round_nearest(penny_bid_price, TICK_SIZE)
                )

                if bid_resp.ok:
                    self.__orders[asset + '_bid'] = (bid_resp.order_id, penny_bid_price, self.size_base)
                    self.__id_to_order[bid_resp.order_id] = asset + '_bid'

                ask_resp = await self.modify_order(
                    old_ask_id,
                    asset,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.ASK,
                    self.size_base,
                    round_nearest(penny_ask_price, TICK_SIZE)
                )

                if ask_resp.ok:
                    self.__orders[asset + '_ask'] = (ask_resp.order_id, penny_ask_price, self.size_base)
                    self.__id_to_order[ask_resp.order_id] = asset + '_ask'
                
                for i in range(0, len(self.level_sizes)):
                    lv = str(i + 1)

                    if (penny_bid_price - self.level_spreads[i]) > 0:
                        old_bid_id, _, __ = self.__orders[asset + 'bid_L' + lv]
                        old_ask_id, _, __ = self.__orders[asset + 'ask_L' + lv]

                        bid_resp = await self.modify_order(
                            old_bid_id,
                            asset,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.BID,
                            self.level_sizes[i],
                            round_nearest(penny_bid_price - self.level_spreads[i], TICK_SIZE)
                        )

                        if bid_resp.ok:
                            self.__orders[asset + '_bid_L' + lv] = (bid_resp.order_id, penny_bid_price - self.level_spreads[i], self.level_sizes[i])
                            self.__id_to_order[bid_resp.order_id] = asset + '_bid_L' + lv

                        ask_resp = await self.modify_order(
                            old_ask_id,
                            asset,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.ASK,
                            self.level_sizes[i],
                            round_nearest(penny_ask_price + self.level_spreads[i], TICK_SIZE)
                        )

                        if ask_resp.ok:
                            self.__orders[asset + '_ask_L' + lv] = (ask_resp.order_id, penny_ask_price + self.level_spreads[i], self.level_sizes[i])
                            self.__id_to_order[ask_resp.order_id] = asset + '_ask_L' + lv


            await asyncio.sleep(0.2)

    async def arbitrage_etf(self): 
        while self._day <= DAYS_IN_YEAR:
            mth = ord('A') + math.floor(self._day / 22)
            nav_ask = 0
            nav_bid = 0
            ask_qty = []
            bid_qty = []
            etf_ask, etf_ask_q = self._best_ask['LLL']
            etf_bid, etf_bid_q = self._best_bid['LLL'] 
            tickers = []
            mult = [5, 3, 2]

            for i in range(mth, mth + 3):
                mult_idx = i - mth
                ticker = 'LBS' + chr(i)
                tickers.append(ticker)

                nav_ask += self._best_ask[ticker][0] * mult_idx
                ask_qty.append(self._best_ask[ticker][1])

                nav_bid += self._best_bid[ticker][0] * mult_idx
                bid_qty.append(self._best_bid[ticker][1])
            
            # if the price to sell etf is higher than the value to buy underlying assets
            # buy underlying assets, create etf, sell etf
            if etf_bid - nav_ask > 0.10:
                etf_qty = min(int(min(ask_qty[0] / 5, ask_qty[1] / 3, ask_qty[2] / 2, etf_bid_q)), 20)

                for i in range(3):
                    id = self.__orders[ticker[i] + '_arb'][0]
                    qty = etf_qty * mult[i]
                    price = self._best_ask[tickers[i]][1]
                    r = await self.modify_order(
                        order_id=id,
                        asset_code=tickers[i],
                        order_type=pb.OrderSpecType.LIMIT,
                        order_side=pb.OrderSpecSide.BID,
                        qty=qty,
                        px=price
                    )

                    if r.ok:
                        self.__orders[ticker[i] + '_arb'] = (r.order_id, price, qty)

                await self.create_etf(etf_qty)

                r_etf = await self.place_order(
                    asset_code='LLL',
                    order_type=pb.OrderSpecType.LIMIT,
                    order_side=pb.OrderSpecSide.ASK,
                    qty=etf_qty,
                    px=self._best_bid['LLL'][1]
                )

                if r_etf.ok:
                    self.__orders['LLL_arb'] = (r_etf.order_id, self._best_bid['LLL'][0], -etf_qty)

            # if the price to buy etf is lower than the price to sell underlying assets
            # buy etf, redeem etf, sell underlying assets
            if nav_bid - etf_ask > 0.10:
                etf_qty = min(int(min(bid_qty[0] / 5, bid_qty[1] / 3, bid_qty[2] / 2, etf_ask_q)), 20)

                r_etf = await self.place_order(
                    asset_code='LLL',
                    order_type=pb.OrderSpecType.LIMIT,
                    order_side=pb.OrderSpecSide.BID,
                    qty=etf_qty,
                    px=self._best_bid['LLL'][1]
                )

                if r_etf.ok:
                    self.__orders['LLL_arb'] = (r_etf.order_id, self._best_bid['LLL'][0], etf_qty)

                await self.redeem_etf(etf_qty)

                for i in range(3):
                    id = self.__orders[ticker[i] + '_arb'][0]
                    qty = etf_qty * mult[i]
                    price = self._best_bid[tickers[i]][1]
                    r = await self.modify_order(
                        order_id=id,
                        asset_code=tickers[i],
                        order_type=pb.OrderSpecType.LIMIT,
                        order_side=pb.OrderSpecSide.ASK,
                        qty=qty,
                        px=price
                    )

                    if r.ok:
                        self.__orders[ticker[i] + '_arb'] = (r.order_id, price, -qty)
            
            await asyncio.sleep(1)

    async def send_bogus_orders(self):
        for asset in CONTRACTS:
            if self._fair_price[asset] == 0:
                await self.calculate_asset_price(asset)
            p = self._fair_price[asset]
            r = await self.place_order(
                asset_code = asset,
                order_type = pb.OrderSpecType.LIMIT,
                order_side = pb.OrderSpecSide.ASK,
                qty = 100,
                px = round_nearest(p + self._spread[asset], TICK_SIZE), 
            )
            if r.ok:
                self.__orders[f'bogus_{asset}_ASK'] = (r.order_id, p + self._spread[asset])

            r = await self.place_order(
                asset_code = asset,
                order_type = pb.OrderSpecType.LIMIT,
                order_side = pb.OrderSpecSide.BID,
                qty = 100,
                px = round_nearest(p - self._spread[asset], TICK_SIZE), 
            )
            if r.ok:
                self.__orders[f'bogus_{asset}_BID'] = (r.order_id, p - self._spread[asset])

        await asyncio.sleep(5) # wait until stuff happens!

    async def handle_read_params(self):
        while True:
            try:
                params = json.load(open(PARAM_FILE, "r"))
                for asset in CONTRACTS:
                    asset_str = asset if asset == 'SBL' or asset == 'LLL' else 'FUT'
                    # self._spread[asset] = params[asset_str]['edge']
                    self._fade[asset] = params[asset_str]['fade']
                    # self._quantity[asset] = params[asset_str]['size']
                    # self._slack[asset] = params[asset_str]['slack']
                    self.size_base = params['size_base']
                    self.level_sizes = params['level_sizes']
                    self.level_spreads = params['level_spreads']
            except:
                print("Unable to read file " + PARAM_FILE)

            await asyncio.sleep(2)
    
    async def print_positions(self):
        while True:
            print("\nDay", self._day) 
            print(self.positions)
            await asyncio.sleep(1)

def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))             

if __name__ == "__main__":
    start_bot(Case1Bot)
