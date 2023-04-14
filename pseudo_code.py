"""
FOR ANU TO-DO
line 24 - change 140 to 135

test to make sure that order maximum is 100. 
It's 100 for futures, but i'm not sure about options so check it out
I'm writing this pseudocode assuming that it's 100.
    
IMPORTANT PARAMETERS
**** penny-in-value for every single option, setas 0.01 for now
(some hotly traded options might require two pennies in
 terrible options like 135C or 65C might be no penny in at all
 because they literally trade for pennies)
 penny-in-value also differs depending on risk limits

**** spread value for every single option
some crappy options like 135C or 65P have bid-ask spreads of cents
others like 65C or 135P have spreads of dollars

**** portfolio risk threshold (from 0 to 1.0)
0.8 means that we are willing to trade up until 0.8 of our greek limit

------------------------------------------------------

at timestamp = 0
    find worst ask and worst bid
    find difference between worst ask/bid and fair price
    penny-in worst ask and worst bid, quantity 100
    repeat for next worst ask and bid
    repeat until halfway between worst bid and ask
    use price differences to send out penny-in deals in opposite direction (they're even worse)


sum greek totals across all current positions
sum greeks for a single option for all options for puts and calls (AKA greek_singles)
i.e. (delta65P+delta70P+...), (delta65C+delta70C+...), (gamma65+gamma70+...)

max_penny_bid_quantity = (greek_limit - greek_total) * / greek_singles_calls
max_penny_ask_quantity = (greek_limit + greek_total) * / -greek_singles_puts
(basically find the quantity you can buy without going over the greek limit assuming
that we are going to buy 1 of every single option in a certain direction
i.e a max call quantity of 5 means we will only send out 5 orders for 65C, 5 orders for 70C, etc.)
there will be 4 different 

if greek_limit hit
    change puts or calls to market orders instead of limits
    continue until halfway between risk_threshold and greek_limit (i.e halfway between 0.8 and 1)

if risk_threshold hit for gamma, theta, vega (i.e. 0.8*greek_limit)
    increase penny-in pricing for puts and calls so that we start buying or selling more options
    (i.e. if penny-in price for selling options is 0.1 and we are too long, increase penny-in-price to 0.2 or higher until below threshold)
if risk_threshold hit for delta
    buy/sell underlying stock
    if delta is above delta risk threhold by 30, find the best bid/ask, buy it all up, and repeat until you are longing/shorting 30 stock
    (delta for stocks are 1)


for each asset
    find best ask that is within BS fair pricing
    find best bid that is within BS fair pricing
    if best ask/bid doesn't exist, we don't trade
    
    find penny-in prices for asks and bids
    send out orders at prices with max quantities 
    
    send out order for 1/2 of max quantity calculated above 
    of ask/bid (50 or lower in this case) at 0.10 better for us for calls and puts
    repeat this until quantity is 0
    
    
    
    
    
    
    
 """