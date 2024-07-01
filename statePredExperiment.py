from experiment import Experiment
from utils.behavior_data import BehaviorData
import torch
import numpy as np
from utils.state_data import StateData
from utils.content import StatesHandler
from models.StateLSTM import StateLSTM
from models.StateNN import StateNN
import argparse
import os


parser = argparse.ArgumentParser()

def toBool(x):
    return (str(x).lower() in ['true', '1', 't'])

# expanded states
parser.add_argument('-model', type=str, default="BasicNN")
parser.add_argument('-numWeeks', type=int, default=30)
parser.add_argument('-estate', type=toBool, default=False)
parser.add_argument('-includeState', type=toBool, default=True)
parser.add_argument('-fullQ', type=toBool, default=False)
parser.add_argument('-insertPreds', type=toBool, default=False)
parser.add_argument('-splitQ', type=toBool, default=False)
parser.add_argument('-splitM', type=toBool, default=False)
parser.add_argument('-catHist', type=toBool, default=False)
parser.add_argument('-epochs', type=int, default=500)
parser.add_argument('-smooth', type=float, default=0)
parser.add_argument('-noise', type=float, default=0.07)
parser.add_argument('-learning_rate', type=float, default=0.001)
parser.add_argument('-respond_perc', type=float, default=0.5)
parser.add_argument('-hierarchical', type=str, default="Shared")
parser.add_argument('-regression', type=toBool, default=False)
parser.add_argument('-nrclass', type=toBool, default=True)
parser.add_argument('-sepHierLoss', type=toBool, default=False)
parser.add_argument('-seeds', type=int, default=5)
parser.add_argument('-only_rnr', type=toBool, default=False)
parser.add_argument('-transformer', type=toBool, default=False)
parser.add_argument('-save', type=toBool, default=True)
parser.add_argument('-cluster_by', type=str, default=None)
parser.add_argument('-num_clusters', type=int, default=3)
parser.add_argument('-cluster_method', type=str, default="Kmeans")



args = parser.parse_args()

if (args.hierarchical != "Shared" and args.hierarchical != "Separate"):
    args.hierarchical = None

model = args.model

respond_perc = args.respond_perc

learning_rate = args.learning_rate

smooth, noise = args.smooth, args.noise

splitQ, splitM = args.splitQ, args.splitM

catHist = args.catHist

numWeeks = args.numWeeks

insertPreds = args.insertPreds

if (args.regression):
    loss_fn = "MSELoss"
else:
    loss_fn = "CrossEntropyLoss"

include_state, estate, fullq = args.includeState, args.estate, args.fullQ

epochs=args.epochs

if "LSTM" in model:
    stateweek = 1
else:
    stateweek = 500

hiddenSize = 50

if (numWeeks > 10):
    lr = .00005
    epochs = epochs
else:
    lr = .0001
    epochs = 2*epochs


bdArgs = {"minw": 2,
    "maxw": 31,
    "include_state": include_state,
    "include_pid": False,
    "expanded_states": estate,
    "top_respond_perc": respond_perc,
    "full_questionnaire": fullq,
    "num_weeks_history": numWeeks,
    "insert_predictions": insertPreds,
    "split_model_features": splitM,
    "split_weekly_questions": splitQ,
    "category_specific_history": catHist,
    "max_state_week": stateweek,
    "regression": args.regression,
    "no_response_class": args.nrclass,
    "only_rnr": args.only_rnr,
    "cluster_by": args.cluster_by,
    "num_clusters": args.num_clusters,
    "cluster_method": args.cluster_method,
    "predictStates": True
    }

for seed in range(args.seeds):
    np.random.seed(seed)
    torch.manual_seed(seed)
    bd = BehaviorData(**bdArgs)
    predictor = StateNN(input_size=bd.dimensions[0], hidden_size=hiddenSize, output_size=bd.dimensions[1])
    # build the optimizer instance
    opt = torch.optim.Adam(params=predictor.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.9)
    for x in range(epochs):
        opt.zero_grad()
        preds = predictor(bd.features[bd.train])
        loss = torch.nn.MSELoss()(preds, bd.labels[bd.train])
        if (x % 10) == 0:
            with torch.no_grad():
                testP = predictor(bd.features[bd.test])
                testLoss = torch.nn.MSELoss()(testP, bd.labels[bd.test])
            # print(loss.item(), testLoss.item())
        loss.backward()
        opt.step()
        sched.step()
    testP = predictor(bd.features[bd.test])
    testLoss = torch.nn.MSELoss()(testP, bd.labels[bd.test])
    print(testLoss.item(), end=", ")
        

    if not args.save:
        continue




