import numpy as np
import matplotlib.pyplot as plt
import os


metrics = ["AccConsumption", "AccKnowledge", "AccExercise"]
metrics = ["Acc"]
w = 2
resp = 0.5
states = 1
estates = 1
full = 0
insertPred = 0
smooth = 0.0
noise = 0.00
splitQ = 0
splitW = 1
chist = 0
maxEpochs = 500

header = ["# MSE","CE","NDCG","MRR","Acc","AccConsumption","AccExercise","AccKnowledge","Acc1","Acc2","Acc3","Prec1","Prec2","Prec3","Rec1","Rec2","Rec3","Count1","Count2","Count3","Week0Acc","Week1Acc","Week2Acc","Week3Acc","Week4Acc","Week5Acc","Week6Acc","Week7Acc","Week8Acc","Week9Acc","Week10Acc","Week11Acc","Week12Acc","Week13Acc","Week14Acc","Week15Acc","Week16Acc","Week17Acc","Week18Acc","Week19Acc","Week20Acc","Week21Acc","Week22Acc","Week23Acc"]

for (model, LR) in [("BasicNN", 0.007)]:
    dir = f"./experiment_output_long/{model}/"
    filename = f"TESTMETRICS-W{w}LR{LR}Resp{resp}States{states}Expanded{estates}Full{full}CHist{chist}Pred{insertPred}Smooth{smooth}Noise{noise}Split{splitQ}{splitW}"
    cols = [header.index(metric) for metric in metrics]
    testresults = []
    for seed in range(500):
        if not os.path.exists(f"{dir}{filename}S{seed}.csv"):
            print(f"{dir}{filename}S{seed}.csv")
            break
        testresults.append(np.loadtxt(f"{dir}{filename}S{seed}.csv", delimiter=",", skiprows=1, usecols=cols))
    testresults = np.stack(testresults, axis=-1)[0:(maxEpochs//5), :]

    testmeans = testresults.mean(axis=-1)
    testdevs = testresults.std(axis=-1)
    testmaxes = testresults.max(axis=-1)
    testmins = testresults.min(axis=-1)


    filename = f"TRAINMETRICS-W{w}LR{LR}Resp{resp}States{states}Expanded{estates}Full{full}CHist{chist}Pred{insertPred}Smooth{smooth}Noise{noise}Split{splitQ}{splitW}"
    cols = [header.index(metric) for metric in metrics]
    results = []
    for seed in range(500):
        if not os.path.exists(f"{dir}{filename}S{seed}.csv"):
            break
        results.append(np.loadtxt(f"{dir}{filename}S{seed}.csv", delimiter=",", skiprows=1, usecols=cols))

    results = np.stack(results, axis=-1)[0:(maxEpochs//5), :]

    trainmeans = results.mean(axis=-1)
    traindevs = results.std(axis=-1)
    trainmaxes = results.max(axis=-1)
    trainmins = results.min(axis=-1)

    x = 5*np.arange(len(testmeans))
    plt.plot(x, testmeans, color="green", label="Test Set")
    plt.fill_between(x, testmins, testmaxes, color=(0, .7, 0, .3))
    plt.plot(x, trainmeans, color="blue", label="Train Set")
    plt.fill_between(x, trainmins, trainmaxes, color=(0, 0, .7, .3))
    plt.legend(fontsize=14, loc="lower right")
    plt.xlabel("Epochs", fontsize=14)
    plt.title("")
    plt.ylabel(f"Accuracy", fontsize=14)
    plt.ylim(bottom=0)
    plt.tick_params(labelsize=14)
    plt.savefig(f"{model}{metrics[0]}.png")
    plt.clf()

