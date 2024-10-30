# script to graph the accuracies of behavior modeling

import numpy as np
import matplotlib.pyplot as plt
import os


metrics = ["AccConsumption", "AccKnowledge", "AccExercise"]
metrics = ["Acc0", "Acc1", "Acc2", "Acc3"]
metrics = ["Acc"]
# metrics = ["# MSE"]
w = 3
resp = 0.5
states = 1
estates = 1
full = 1
insertPred = 1
smooth = 0.0
noise = 0.05
splitQ = 1
splitW = 1
chist = 0
maxEpochs = 1000
reg = 0
nrclass = True
hier = "None"
# hier = "Separate"
hier = "Shared"
sharedHierLoss = 0
cluster = "Demographics"
nc = 3
cmethod = "Gaussian"

for (model, LR) in [("AdaptableLSTM", 0.002), ("BasicNN", .0025)]:
    dir = f"./experiment_output_long/{model}Attn/cluster{cluster}/"
    filePrefix = f"C{nc}{cmethod}R{reg}NR{nrclass}H{hier}{sharedHierLoss}W{w}LR{LR}Resp{resp}States{states}Expanded{estates}Full{full}CHist{chist}Pred{insertPred}Smooth{smooth}Noise{noise}Split{splitQ}{splitW}"
    filename = f"TESTMETRICS-{filePrefix}"
    header = np.loadtxt(f"{dir}{filename}S0.csv", delimiter=",", skiprows=0, max_rows=1, dtype=str, comments=None).tolist()
    print(header)
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


    filename = f"TRAINMETRICS-{filePrefix}"
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

