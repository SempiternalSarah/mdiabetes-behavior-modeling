import numpy as np
import matplotlib.pyplot as plt
import os


# metrics = ["AccConsumption", "AccKnowledge", "AccExercise"]
# metrics = ["Acc", "Acc0", "Acc1", "Acc2", "Acc3"]
# metrics = ["Acc", "AccRes"]
metrics = ["Acc"]
w = 3
resp = 0.5
states = 1
estates = 1
full = 1
insertPred = 0
smooth = 0.0
noise = 0.05
splitQ = 1
splitW = 1
chist = 0
reg = 0
nrclass = True
sepRegLoss = 0

#/FINALTESTMETRICS-W30LR0.005Resp0.5States1Expanded1Full1CHist1Pred1Smooth0.0Noise0.07Split11.csv
for nc in [2, 3, 4, 5]:
    for cmethod in ["Gaussian", "Kmeans", "Spectral"]:
        for (model, LR) in [("BasicNN", 0.00025),("AdaptableLSTM", 0.0002)]:
            for cluster in ["Initial", "Demographics"]:
            # for (model, LR) in [("AdaptableLSTM", 0.07)]:
            # for (model, LR) in [("BasicNN", 0.003)]:
                for hier in ["None", "Shared"]:
                    for splitW in [0, 1]:
                        # for dataset in ["TRAIN", "TEST"]:
                        for dataset in ["TEST"]:
                            dir = f"./experiment_output/{model}Attn/cluster{cluster}/"
                            filename = f"FINAL{dataset}METRICS-C{nc}{cmethod}R{reg}NR{nrclass}H{hier}{sepRegLoss}W{w}LR{LR}Resp{resp}States{states}Expanded{estates}Full{full}CHist{chist}Pred{insertPred}Smooth{smooth}Noise{noise}Split{splitQ}{splitW}"
                            if not os.path.exists(f"{dir}{filename[5:]}S0.csv"):
                                print(f"{dir}{filename[5:]}S0.csv")
                                continue
                            header = np.loadtxt(f"{dir}{filename[5:]}S0.csv", delimiter=",", skiprows=0, max_rows=1, dtype=str, comments=None).tolist()
                            cols = [header.index(metric) for metric in metrics]
                            results = np.loadtxt(f"{dir}{filename}.csv", delimiter=",", skiprows=0, usecols=cols)
                            means = results.mean(axis=0)
                            devs = results.std(axis=0)
                            prstr = ""
                            if (type(means) is np.ndarray):
                                for metric in range(len(means)):
                                    # prstr += f" & {means[metric]:.3}±{devs[metric]:03.3f}"
                                    prstr += f"{means[metric]:.3}±{devs[metric]:03.3f}, "
                                prstr = prstr[:-2]
                            else:
                                # prstr += f" & {means:.3}±{devs:03.3f}"
                                # prstr += f"{means:.3}±{devs:03.3f}"
                                prstr += f"{means:.3}, {devs:03.3f}"
                            if (hier == "Shared"):
                                hstring = "Shared Hierarchical"
                            elif (hier == "Separate"):
                                hstring = "Separate Hierarchical"
                            else:
                                hstring = "Flat"
                            if (splitW):
                                tstring = "Type Trifecta"
                            else:
                                tstring = "Single Model"
                            cstring = f"{nc} {cluster} by {cmethod}"
                            # print(f"{model}, {hstring} {tstring} {dataset} {prstr}")
                            print(f"{model}, {cstring}, {hstring}, {tstring}, {dataset}, {prstr}")
                                

