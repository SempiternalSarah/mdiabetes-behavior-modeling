import torch.nn as nn
from torch.autograd import Variable
import torch
import torchmetrics
from models.ModelUtils import *
import numpy as np

# our categories
RESP_CATEGORIES = [0, 1, 2, 3]
N_RESP_CATEGORIES = len(RESP_CATEGORIES)

# base class for models, mostly just utility code, not a real model
class Base(nn.Module):
    
    def __init__(self, 
                 input_size=36, hidden_size=256, output_size=6,
                 lossfn="CrossEntropyLoss", loss_kw={},
                 lr_step_mult=.9, lr_step_epochs=60,
                 optimizer="Adam", opt_kw={"lr": 1e-3},
                 labelSmoothPerc = 0.0, gaussianNoiseStd = 0.0,
                 numTimesteps = 24,
                 splitModel = False,
                 splitWeeklyQuestions = False):
        # define all inputs to the model
        # input_size:   # features in input
        # hidden_size:  # size of hidden layer
        # output_size   # features in output
        # lossfn:       # string representing torch loss fn
        # loss_kw:      # keyword arguments to loss fn
        # optimizer:    # string represneting torch optimizer
        # opt_kw:       # keyword arguments to optimizer
        super().__init__()
        self.lr_step_mult = lr_step_mult
        self.lr_step_epochs = lr_step_epochs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lossfn, self.loss_kw = lossfn, loss_kw
        self.optimizer, self.opt_kw = optimizer, opt_kw
        self.labelSmoothPerc = labelSmoothPerc
        self.gaussianNoiseStd = gaussianNoiseStd
        self.numTimesteps = numTimesteps
        self.splitModel = splitModel
        self.splitWeeklyQuestions = splitWeeklyQuestions
        
        
    def forward(self, x):
        # fake forward function so we can do other stuff
        return x
    
    def predict(self, x):
        # call forward method but do not collect gradient
        with torch.no_grad():
            return self.forward(x)
        
    def make_criterion(self):
        # build the loss function
        return getattr(torch.nn, self.lossfn)(**self.loss_kw)
    
    def make_optimizer(self):
        # build the optimizer instance
        optcls = getattr(torch.optim, self.optimizer)
        opt = optcls(self.parameters(), **self.opt_kw)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=self.lr_step_epochs, gamma=self.lr_step_mult)
        return opt, sched
        
    def _init_hc(self):
        # initialize the hidden/cell state variables
        h0 = Variable(torch.zeros(1, self.hidden_size))
        c0 = Variable(torch.zeros(1, self.hidden_size))
        return h0, c0

    def train_step(self, pred, y, data, trainConsumption=True, trainKnowledge=True, trainPhys=True):
        # One optimization step of our model on 
        # predictions pred with labels y
        # make predictions using forward() before calling this function
        if (self.lossfn == "NDCG"):
            crit = NDCG
        else:
            crit = self.make_criterion()
        # separate out category labels
        if (self.splitWeeklyQuestions):
            data = data[:, -2:]
        else:
            data = data[:, -4:]
        if self.splitWeeklyQuestions or self.splitModel:
            k = self.output_size
        else:
            k = self.output_size // 2
        if not self.splitWeeklyQuestions:
            pred = pred.view((y.shape[0], k*2))
            # reshape preds/labels so that one question = one row
            pred = torch.cat([pred[:, :k], pred[:, k:]], 0)
            y = torch.cat([y[:, :k + 1], y[:, k + 1:]], 0)
            data = torch.cat([data[:, 0:2], data[:, 2:]], 0)
        # calculate loss, ignoring nonresponses
        pred = pred[y[:, 0] != 1]
        data = data[y[:, 0] != 1]
        y = y[y[:, 0] != 1]
        y = y[:, 1:]
        if (self.splitModel):
            consumptionRows = (torch.where(data[:, -1] == 0, 1, 0) * torch.where(data[:, -2] == 0, 1, 0)).nonzero()
            knowledgeRows = (torch.where(data[:, -1] == 0, 1, 0) * torch.where(data[:, -2] == 1, 1, 0)).nonzero()
            physRows = (torch.where(data[:, -1] == 1, 1, 0) * torch.where(data[:, -2] == 0, 1, 0)).nonzero()
            mask = np.ones(data.shape[0], dtype=bool)
            if not trainConsumption:
                mask[consumptionRows] = False
            if not trainKnowledge:
                mask[knowledgeRows] = False
            if not trainPhys:
                mask[physRows] = False
            pred = pred[mask]
            y = y[mask] 

        if (pred.numel() <= 0):
            return None

        # smooth labels for hopefully better overall results
        l = self.labelSmoothPerc
        if (l > 0):
            y[y[:, 0] == 1] += torch.tensor([[-l, l, 0]])
            y[y[:, 1] == 1] += torch.tensor([[2*l/3, -4*l/3, 2*l/3]])
            y[y[:, 2] == 1] += torch.tensor([[0, l, -l]])
        g = self.gaussianNoiseStd
        if (g > 0):
            y += torch.normal(mean=torch.zeros_like(y), std=g * torch.ones_like(y))
            # re-normalize labels to ensure no < 0 and that sum = 1
            y[y < 0] = 0
            # divide each row by sum of that row
            y /= y.sum(dim=1).unsqueeze(-1).expand(y.size())

        loss = crit(pred, y)
        return loss
            

    def report_scores_min(self, y, pred, data):
        if self.splitWeeklyQuestions or self.splitModel:
            k = self.output_size
        else:
            k = self.output_size // 2

        # separate out category labels
        if (self.splitWeeklyQuestions):
            data = data[:, -2:]
        else:
            data = data[:, -4:]
        # reshape so 1 response = 1 row if needed
        if not self.splitWeeklyQuestions:
            pred = torch.cat([pred[:, :k], pred[:, k:]], 0)
            y = torch.cat([y[:, :k + 1], y[:, k + 1:]], 0)
            data = torch.cat([data[:, 0:2], data[:, 2:]], 0)
        # calculate loss, ignoring nonresponses
        pred = pred[y[:, 0] != 1]
        data = data[y[:, 0] != 1]
        y = y[y[:, 0] != 1]
        if (y.numel() > 0):
            crit = torch.nn.MSELoss()
            mseloss = crit(pred, y[:, 1:])
            crit = torch.nn.CrossEntropyLoss()
            celoss = crit(pred, y[:, 1:])
            crit = NDCG
            ndcg = crit(pred, y[:, 1:])
            crit = MRR
            mrr = crit(pred, y[:, 1:])
            sm = torch.nn.Softmax(dim=1)
            predValues = torch.nn.functional.one_hot(torch.argmax(pred, dim=1), num_classes=k)
            class_precision, class_recall = torchmetrics.functional.precision_recall(pred.argmax(dim=1), y[:, 1:].argmax(dim=1), average='none', num_classes=k)
            accuracy = (predValues * y[:, 1:]).sum() / y.shape[0]
            # filter predicted classes by true class
            pred1 = predValues[y[:, 1] == 1]
            pred2 = predValues[y[:, 2] == 1]
            pred3 = predValues[y[:, 3] == 1]
            # per class accuracy
            acc1 = (pred1[:, 0].sum()) / pred1.shape[0]
            acc2 = (pred2[:, 1].sum()) / pred2.shape[0]
            acc3 = (pred3[:, 2].sum()) / pred3.shape[0]

            # record per category accuracy
            if (self.splitModel):
                classLabels = ["AccConsumption", "AccExercise", "AccKnowledge"]
                consumptionRows = (torch.where(data[:, -1] == 0, 1, 0) * torch.where(data[:, -2] == 0, 1, 0)).nonzero()
                knowledgeRows = (torch.where(data[:, -1] == 0, 1, 0) * torch.where(data[:, -2] == 1, 1, 0)).nonzero()
                physRows = (torch.where(data[:, -1] == 1, 1, 0) * torch.where(data[:, -2] == 0, 1, 0)).nonzero()
                conRows, conYs = predValues[consumptionRows], y[consumptionRows, 1:]
                exerRows, exerYs = predValues[physRows], y[physRows, 1:]
                knowRows, knowYs = predValues[knowledgeRows], y[knowledgeRows, 1:]
                consumptionAcc = (conRows * conYs).sum() / conRows.shape[0]
                exerciseAcc = (exerRows * exerYs).sum() / exerRows.shape[0]
                knowledgeAcc = (knowRows * knowYs).sum() / knowRows.shape[0]
                classStats = [consumptionAcc, exerciseAcc, knowledgeAcc]
            else:
                classLabels = []
                classStats = []
                
            # per timestep accuracy
            timeAcc = []
            timeLabels = []
            for x in range(self.numTimesteps):
                indices = np.arange(x, predValues.shape[0], self.numTimesteps)
                timeAcc.append((predValues[indices] * y[indices, 1:]).sum() / len(indices))
                timeLabels.append(f"Week{x}Acc")
            return np.array([mseloss.item(), celoss.item(), ndcg.item(), mrr.item(), accuracy.item()] + classStats + [acc1, acc2, acc3, class_precision[0].item(), class_precision[1].item(), class_precision[2].item(), class_recall[0].item(), class_recall[1].item(), class_recall[2].item(), pred1.shape[0], pred2.shape[0], pred3.shape[0]] + timeAcc), ["MSE", "CE", "NDCG", "MRR", "Acc"] + classLabels + ["Acc1", "Acc2", "Acc3", "Prec1", "Prec2", "Prec3", "Rec1", "Rec2", "Rec3", "Count1", "Count2", "Count3"] + timeLabels
        else:
            return [], ["MSE", "CE", "NDCG", "MRR", "Acc", "ResCount"]
                     
                     
            