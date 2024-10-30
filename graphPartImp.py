import numpy as np
import matplotlib.pyplot as plt
import os

# envs = ["Humanoid-v2"]
env = "Diabetes"
seedmin, seedmax = 1, 40
hiddens = [1]
statepreds = ['', 'lstm']
variables = ['totalimp']
maxEpoch = 200
qlr = 3e-5
hsize = 128
eqs = ["True"]


np.random.seed(0)

colors = plt.cm.rainbow(np.linspace(0, 1, len(hiddens) * len(statepreds)))
for eq in eqs:
    for v, variable in enumerate(variables):
        for s, statepred in enumerate(statepreds):
            temp = None
            folder = "./saved_mdiabetes_rl"
            for i, hidden in enumerate(hiddens):
                if statepred == '':
                    folder = f"{folder}/trajlog"
                else:
                    folder = f"{folder}/{statepred}/trajlog"
                for seed in range(seedmin, seedmax+1):
                    fname = f"{folder}/{variable}{seed}H{hsize}LR{qlr}EQ{eq}.csv"
                    if not os.path.isfile(fname):
                        print(f"NOT FOUND: {fname}")
                        continue
                    if temp is None:
                        temp = np.expand_dims(np.loadtxt(fname, delimiter="\n"), axis=-1)
                        if len(temp) > 200:
                            temp = temp[199:199+maxEpoch]
                        else:
                            temp = temp[:maxEpoch]
                    else:
                        tempin = np.expand_dims(np.loadtxt(fname, delimiter="\n"), axis=-1)
                        if len(tempin) > 200:
                            tempin = tempin[199:199+maxEpoch]
                        else:
                            tempin = tempin[:maxEpoch]
                        if tempin.shape[0] != temp.shape[0]:
                            print(f"{fname} unused")
                            continue
                        temp = np.concatenate([temp, tempin], axis=-1)
                if len(temp.shape) < 2:
                    temp = temp.expand_dims(-1)
                testmeans = np.mean(temp, axis=-1)
                stddev = np.std(temp, axis=-1)
                x = np.arange(len(testmeans)) / 200
                testmins = testmeans - stddev
                testmaxes = testmeans + stddev
                if statepred == '':
                    label = f"RRD"
                else:
                    label = f"SPER2D (LSTM)"
                print(f"{env}- {label}\n\t{(testmeans[-1] * 100):.2f}±{(stddev[-1] * 100/np.sqrt(40)):.2f}")
                plt.plot(x, testmeans, color=colors[s*len(hiddens) + i], label=label)
                plt.fill_between(x, testmins, testmaxes, color=colors[s*len(hiddens) + i], alpha=0.1)

    plt.title(f"Diabetes Messaging", fontsize=22)
    plt.ylabel("% of Participants Improved", fontsize=20)
    plt.xlabel("Environmental Steps (millions)", fontsize=20)
    plt.ylim(bottom=0)
    plt.legend(fontsize=18, loc="lower right")
    plt.tight_layout()
    print(f"./MJGraphs/{variable}EQ{eq}form.png")
    plt.savefig(f"./MJGraphs/{variable}EQ{eq}form.png")
    plt.clf()
                
                
    # plt.title(f"Diabetes Messaging", fontsize=22)
    # plt.ylabel("State Pred Loss", fontsize=20)
    # plt.xlabel("Epochs (millions)", fontsize=20)
    # plt.legend(fontsize=18, loc="lower right")
    # plt.tight_layout()
    # plt.savefig(f"./MJGraphs/StateLossMdbRl.png")
    # plt.clf()



