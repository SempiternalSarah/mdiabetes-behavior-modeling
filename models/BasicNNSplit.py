from torch import nn
import numpy as np
import torch
from models.ModelUtils import *
from models.base import Base, N_RESP_CATEGORIES

class BasicNNSplit(Base):
    
    def __init__(self, *args, **kw):
        # Simple model: lstm block with one linear output layer
        # input_size:  size of linear input to model
        # hidden_size: size of lstm hidden layer
        # output_size: size of output label of data
        super().__init__(*args, **kw)
        self.inputLayer = nn.Linear(self.input_size, self.hidden_size)
        self.fc_q1 = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        output = self.inputLayer(x) 
        out = self.relu(output)
        return self.fc_q1(out).softmax(-1)
