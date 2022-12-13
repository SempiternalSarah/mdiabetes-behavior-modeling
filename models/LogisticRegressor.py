from torch import nn
import numpy as np
import torch
from models.ModelUtils import *
from models.base import Base, N_RESP_CATEGORIES

class LogisticRegressor(Base):
    
    def __init__(self, *args, **kw):
        # Simple model: lstm block with two linear output layers,
        #    one to predict the response of either label (question)
        # input_size:  size of linear input to model
        # hidden_size: size of lstm hidden layer
        # output_size: size of output label of data
        super().__init__(*args, **kw)
        self.linear = nn.Linear(self.input_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
       return self.sigmoid(self.linear(x))