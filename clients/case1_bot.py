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
import torch
import torch.nn as nn
import torch.optim as optim

DAYS_IN_MONTH = 21
DAYS_IN_YEAR = 252
INTEREST_RATE = 0.02
NUM_FUTURES = 14
TICK_SIZE = 0.00001
FUTURE_CODES = [chr(ord('A') + i) for i in range(NUM_FUTURES)] # Suffix of monthly future code
CONTRACTS = ['SBL'] +  ['LBS' + c for c in FUTURE_CODES] + ['LLL']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUModel(nn.Module):
    """GRUModel class extends nn.Module class and works as a constructor for GRUs.

       GRUModel class initiates a GRU module based on PyTorch's nn.Module class.
       It has only two methods, namely init() and forward(). While the init()
       method initiates the model with the given input parameters, the forward()
       method defines how the forward propagation needs to be calculated.
       Since PyTorch automatically defines back propagation, there is no need
       to define back propagation method.

       Attributes:
           hidden_dim (int): The number of nodes in each layer
           layer_dim (str): The number of layers in the network
           gru (nn.GRU): The GRU model constructed with the input parameters.
           fc (nn.Linear): The fully connected layer to convert the final state
                           of GRUs to our desired output shape.

    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates a GRU instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

class Optimization:
    """Optimization is a helper class that allows training, validation, prediction.

    Optimization is a helper class that takes model, loss function, optimizer function
    learning scheduler (optional), early stopping (optional) as inputs. In return, it
    provides a framework to train and validate the models, and to predict future values
    based on the models.

    Attributes:
        model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
        loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
        optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        train_losses (list[float]): The loss values from the training
        val_losses (list[float]): The loss values from the validation
        last_epoch (int): The number of epochs that the models is trained
    """
    def __init__(self, model, loss_fn, optimizer):
        """
        Args:
            model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
            loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
            optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        
    def train_step(self, x, y):
        """The method train_step completes one step of training.

        Given the features (x) and the target values (y) tensors, the method completes
        one step of the training. First, it activates the train mode to enable back prop.
        After generating predicted values (yhat) by doing forward propagation, it calculates
        the losses by using the loss function. Then, it computes the gradients by doing
        back propagation and updates the weights by calling step() function.

        Args:
            x (torch.Tensor): Tensor for features to train one step
            y (torch.Tensor): Tensor for target values to calculate losses

        """
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        """The method train performs the model training

        The method takes DataLoaders for training and validation datasets, batch size for
        mini-batch training, number of epochs to train, and number of features as inputs.
        Then, it carries out the training by iteratively calling the method train_step for
        n_epochs times. If early stopping is enabled, then it  checks the stopping condition
        to decide whether the training needs to halt before n_epochs steps. Finally, it saves
        the model in a designated file path.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader that stores training data
            val_loader (torch.utils.data.DataLoader): DataLoader that stores validation data
            batch_size (int): Batch size for mini-batch training
            n_epochs (int): Number of epochs, i.e., train steps, to train
            n_features (int): Number of feature columns

        """
        model_path = f'./models/gru.pt'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        """The method evaluate performs the model evaluation

        The method takes DataLoaders for the test dataset, batch size for mini-batch testing,
        and number of features as inputs. Similar to the model validation, it iteratively
        predicts the target values and calculates losses. Then, it returns two lists that
        hold the predictions and the actual values.

        Note:
            This method assumes that the prediction from the previous step is available at
            the time of the prediction, and only does one-step prediction into the future.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
            batch_size (int): Batch size for mini-batch training
            n_features (int): Number of feature columns

        Returns:
            list[float]: The values predicted by the model
            list[float]: The actual values in the test set.

        """
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

# trained GRU model for price prediction
gru = torch.load('../case1/analysis/models/gru.pt')
# load into optimization helper
loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model=gru.parameters(), lr=1e-3, weight_decay=1e-6)
opt = Optimization(model=gru, loss_fn=loss_fn, optimizer=optimizer)

class Case1Bot(UTCBot):
    etf_suffix = ''
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
    
    async def days_to_expiry(self, asset):
        '''
        Calculates days to expiry for the future
        '''
        future = ord(asset[-1]) - ord('A')
        expiry = 21 * (future + 1)
        return self._day - expiry

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
                self._best_ask[asset] = float(book.bids[0].px)
            


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
        asyncio.create_task(self.example_redeem_etf())
        
        # Starts market making for each asset
        # for asset in CONTRACTS:
            # asyncio.create_task(self.make_market_asset(asset))

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
    
    async def calculate_fair_price(self, asset):
        pass
        
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

def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))             

if __name__ == "__main__":
    start_bot(Case1Bot)
