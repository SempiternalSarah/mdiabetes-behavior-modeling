from experiment import Experiment
from utils.behavior_data import BehaviorData
from visuals import Plotter
import torch
import numpy as np
from utils.state_data import StateData

np.random.seed(0)
torch.manual_seed(0)
e = Experiment(
    data_kw={"minw": 2,
            "maxw": 27,
            "include_state": True,
            "include_pid": False},
    model="AdaptableLSTM",
    # model="BasicNN",
    model_kw={
        "lossfn": "MSELoss",
        # "lossfn": "NDCG",
        # "lossfn": "CrossEntropyLoss",
        "hidden_size": 200, 
        "opt_kw": {
            "lr": 1
        }
    },
    train_kw={
        "epochs": 100,
        "n_subj": 500,
        "rec_every": 2,
    })

report = e.run()

metrics, labels = e.report_scores()
for idx, l in enumerate(labels):
    print(metrics[idx])


Plotter.training_loss(report)