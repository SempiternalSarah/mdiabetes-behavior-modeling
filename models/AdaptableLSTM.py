from torch import nn
import numpy as np
import torch
from models.base import Base, N_RESP_CATEGORIES
from models.ModelUtils import *

class AdaptableLSTM(Base):
    
    def __init__(self, *args, **kw):
        # Simple model: lstm block with two linear output layers,
        #    one to predict the response of either label (question)
        # input_size:  size of linear input to model
        # hidden_size: size of lstm hidden layer
        # output_size: size of output label of data
        super().__init__(*args, **kw)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.fc_q1 = nn.Linear(self.hidden_size, self.output_size//2)
        self.fc_q2 = nn.Linear(self.hidden_size, self.output_size//2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # One forward pass of input vector/batch x
        # Pass x thru the LSTM cell, then pass output
        #    through each linear output layer for each
        #    response prediction. 
        # Then join predictions together as one vector
        H0, C0 = self._init_hc()
        output, (H,C) = self.lstm(x, (H0, C0))
        # print(H, output)
        # [SEQ, hidden_size]
        
        out = self.relu(output)
        out_q1 = self.fc_q1(out).softmax(-1)
        # [SEQ, output_size//2]
        out_q2 = self.fc_q2(out).softmax(-1)
        # [SEQ, output_size//2]
        return torch.cat([out_q1, out_q2],-1)
    
    def w1_reg(self, y):
        k = self.fc_q1.out_features
        y1 = y[:, :k]
        y2 = y[:, k:]
        t = y.shape[0]
        if (t < 2):
            return 0
        sum1 = torch.abs(y1[1:] - y1[:-1]).sum()
        sum2 = torch.abs(y2[1:] - y2[:-1]).sum()
        w1 = (sum1 + sum2) / (2 * (t-1))
        return w1
    
    def w2_reg(self, y):
        k = self.fc_q1.out_features
        y1 = y[:, :k]
        y2 = y[:, k:]
        t = y.shape[0]
        if (t < 2):
            return 0
        sum1 = torch.sqrt((y1[1:] - y1[:-1]) ** 2).sum()
        sum2 = torch.sqrt((y2[1:] - y2[:-1]) ** 2).sum()
        w2 = (sum1 + sum2) / (2 * (t-1))
        return w2
        