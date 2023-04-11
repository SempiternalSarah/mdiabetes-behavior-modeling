import numpy as np
import matplotlib.pyplot as plt
import os


# metrics = ["Acc", "AccConsumption", "AccKnowledge", "AccExercise"]
metrics = ["Acc"]
w = 2
resp = 0.5
states = 1
estates = 1
insertPred = 1
smooth = 0.0
noise = 0.07
splitQ = 1
splitM = 0
fullQ = 0
catHist = 0
dataset = "TRAIN"
# dataset = "TEST"

#/FINALTESTMETRICS-W30LR0.005Resp0.5States1Expanded1Full1CHist1Pred1Smooth0.0Noise0.07Split11.csv
header = ["# MSE","CE","NDCG","MRR","Acc","AccConsumption","AccExercise","AccKnowledge","Acc1","Acc2","Acc3","Prec1","Prec2","Prec3","Rec1","Rec2","Rec3","Count1","Count2","Count3","Week0Acc","Week1Acc","Week2Acc","Week3Acc","Week4Acc","Week5Acc","Week6Acc","Week7Acc","Week8Acc","Week9Acc","Week10Acc","Week11Acc","Week12Acc","Week13Acc","Week14Acc","Week15Acc","Week16Acc","Week17Acc","Week18Acc","Week19Acc","Week20Acc","Week21Acc","Week22Acc","Week23Acc"]

for (model, LR) in [("BasicNN", 0.003),("AdaptableLSTM", 0.005),("LogisticRegressor", 0.001)]:
# for (model, LR) in [("BasicNN", 0.003)]:
    dir = f"./experiment_output/{model}/"
    filename = f"FINAL{dataset}METRICS-W{w}LR{LR}Resp{resp}States{states}Expanded{estates}Full{fullQ}CHist{catHist}Pred{insertPred}Smooth{smooth}Noise{noise}Split{splitQ}{splitM}"
    cols = [header.index(metric) for metric in metrics]
    results = np.loadtxt(f"{dir}{filename}.csv", delimiter=",", skiprows=0, usecols=cols)
    means = results.mean(axis=0)
    devs = results.std(axis=0)
    prstr = ""
    if (type(means) is np.ndarray):
        for metric in range(len(means)):
            prstr += f" & {means[metric]:.3}±{devs[metric]:03.3f}"
    else:
        prstr += f" & {means:.3}±{devs:03.3f}"
    print(f"{model}, 3 Models {prstr}\\\\")
        

