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
import orjson
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

SIZE_BASE = 25
LEVEL_SIZES = [10, 5, 5]
LEVEL_SPREADS = [0.05, 0.15, 0.25]

class Case1Bot(UTCBot):
    etf_suffix = ''

    soy_prices = np.zeros(DAYS_IN_YEAR)
    soy_prices[0] = 53.803717019730705

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
                # await self.calculate_soy_price()

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
                self._best_bid[asset] = float(best_bid.px) if best_bid is not None else 0
                self._best_ask[asset] = float(best_ask.px) if best_ask is not None else 0
            
            self.soy_prices[self._day] = (self._best_bid['SBL'] + self._best_ask['SBL']) / 2

        elif kind == "pnl_msg":
            print('Realized pnl:', update.pnl_msg.realized_pnl, "| M2M pnl:", update.pnl_msg.m2m_pnl)

        # elif kind == "fill_msg":
            # print(update.fill_msg.order_id, "filled:", "BUY" if update.fill_msg.order_side == "BUY" else "SELL", 
                #   update.fill_msg.asset, "@", update.fill_msg.price, "x", update.fill_msg.filled_qty)
            # order = self.__order_id_map[update.fill_msg.order_id]
            # self.__orders[] = ("", 0)

        elif kind == "order_cancelled_msg":
            ids = update.order_cancelled_msg.order_ids
            print(ids, "cancelled:", update.order_cancelled_msg.asset, 
                  "intentional!" if update.order_cancelled_msg.intentional else "not intentional!")
            # for id in ids:
                # order = self.__order_id_map[id]
                # del self.__orders[order]
            # for id in ids:
                # self.__orders[id] = ("", 0)
                # pass
            

    async def handle_round_started(self):
        ### Current day
        self._day = 0
        ### Best Bid in the order book
        self._best_bid: Dict[str, float] = defaultdict(lambda: 0)
        ### Best Ask in the order book
        self._best_ask: Dict[str, float] = defaultdict(lambda: 0)
        ### Order book for market making
        self.__orders: DefaultDict[str, Tuple[str, float]] = defaultdict(lambda: ("", 0))
        self.__order_id_map: DefaultDict[str, str] = defaultdict(lambda: "")
        ### TODO Recording fair price for each asset
        self._fair_price: DefaultDict[str, float] = defaultdict(lambda: 0)
        ### TODO spread fair price for each asset
        self._spread: DefaultDict[str, float] = defaultdict(lambda: 0)
        self._fade: DefaultDict[str, float] = defaultdict(lambda: 0)
        self._slack: DefaultDict[str, float] = defaultdict(lambda: 0)
        ### TODO order size for market making positions
        self._quantity: DefaultDict[str, int] = defaultdict(lambda: 0)
        
        ### List of weather reports
        self._weather_log = []
        
        await asyncio.sleep(.1)
        """
        STARTING ASYNC FUNCTIONS
        """
        asyncio.create_task(self.handle_read_params())
        # asyncio.create_task(self.send_bogus_orders())
        asyncio.create_task(self.print_positions())
        
        # Starts market making for each asset
        for asset in CONTRACTS:
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
            if asset != 'SBL' and asset != 'LLL' and self._day >= 22 * (ord('A') - ord(asset[-1]) + 1):
                print(self._day, asset)
                # skip futures that have already expired!
                return

            await self.calculate_asset_price(asset)

            penny_ask_price = self._best_ask[asset] - 0.01 - self._fade[asset] * self.positions.get(asset, 0) / 100
            penny_bid_price = self._best_bid[asset] + 0.01 + self._fade[asset] * self.positions.get(asset, 0) / 100

            if penny_ask_price - penny_bid_price > 0:
                old_bid_id, _ = self.__orders[asset + '_bid']
                old_ask_id, _ = self.__orders[asset + '_ask']

                bid_resp = await self.modify_order(
                    old_bid_id,
                    asset,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.BID,
                    SIZE_BASE,
                    round_nearest(penny_bid_price, TICK_SIZE)
                )

                if bid_resp.ok:
                    self.__orders[asset + '_bid'] = (str(penny_bid_price), bid_resp.order_id)
                    # self.__order_id_map[bid_resp.order_id] = asset + '_bid'

                ask_resp = await self.modify_order(
                    old_ask_id,
                    asset,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.ASK,
                    SIZE_BASE,
                    round_nearest(penny_ask_price, TICK_SIZE)
                )

                if ask_resp.ok:
                    self.__orders[asset + '_ask'] = (str(penny_ask_price), ask_resp.order_id)
                    # self.__order_id_map[ask_resp.order_id] = asset + '_ask'
                
                for i in range(0, len(LEVEL_SIZES)):
                    lv = str(i + 1)

                    if (penny_bid_price - LEVEL_SPREADS[i]) > 0:
                        old_bid_id, _ = self.__orders[asset + 'bid_L' + lv]
                        old_ask_id, _ = self.__orders[asset + 'ask_L' + lv]

                        bid_resp = await self.modify_order(
                            old_bid_id,
                            asset,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.BID,
                            LEVEL_SIZES[i],
                            round_nearest(penny_bid_price - LEVEL_SPREADS[i], TICK_SIZE)
                        )

                        if bid_resp.ok:
                            self.__orders[asset + '_bid_L' + lv] = (str(penny_bid_price - LEVEL_SPREADS[i]), bid_resp.order_id)
                            # self.__order_id_map[bid_resp.order_id] = asset + '_bid' + lv

                        ask_resp = await self.modify_order(
                            old_ask_id,
                            asset,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.ASK,
                            LEVEL_SIZES[i],
                            round_nearest(penny_ask_price + LEVEL_SPREADS[i], TICK_SIZE)
                        )

                        if ask_resp.ok:
                            self.__orders[asset + '_ask_L' + lv] = (str(penny_ask_price + LEVEL_SPREADS[i]), ask_resp.order_id)
                            # self.__order_id_map[bid_resp.order_id] = asset + '_ask' + lv


            await asyncio.sleep(0.1)

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

    async def handle_read_params(self):
        while True:
            try:
                params = orjson.loads(open(PARAM_FILE, "r").read())
                for asset in CONTRACTS:
                    asset_str = asset if asset == 'SBL' or asset == 'LLL' else 'FUT'
                    self._spread[asset] = params[asset_str]['edge']
                    self._fade[asset] = params[asset_str]['fade']
                    # self._quantity[asset] = params[asset_str]['size']
                    # self._slack[asset] = params[asset_str]['slack']
            except:
                print("Unable to read file " + PARAM_FILE)

            await asyncio.sleep(5)
    
    async def print_positions(self):
        while True:
            print("\nDay", self._day)
            print("Positions:", self.positions)
            await asyncio.sleep(1)

def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))             

if __name__ == "__main__":
    start_bot(Case1Bot)
