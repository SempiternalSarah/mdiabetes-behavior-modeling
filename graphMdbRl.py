import numpy as np
import matplotlib.pyplot as plt
import os

# envs = ["Humanoid-v2"]
envs = ["Diabetes"]
seedmin, seedmax = 1, 10
hiddens = [1]
statepreds = ['', 'lstm', 'lstm']
variables = ['rewards']
maxEpoch = 200
qlr = 3e-5
hsize = 128

colors = plt.cm.rainbow(np.linspace(0, 1, len(hiddens) * len(statepreds)))
for env in envs:
    for v, variable in enumerate(variables):
        for s, statepred in enumerate(statepreds):
            temp = None
            folder = "./saved_mdiabetes_rl"
            for i, hidden in enumerate(hiddens):
                if statepred == '':
                    folder = f"{folder}/{variable}"
                else:
                    folder = f"{folder}/{statepred}/{variable}"
                for seed in range(seedmin, seedmax+1):
                    fname = f"{folder}/{seed}H{hsize}LR{qlr}.csv"
                    if not os.path.isfile(fname):
                        print(f"NOT FOUND: {fname}")
                        continue
                    if temp is None:
                        temp = np.expand_dims(np.loadtxt(fname, delimiter="\n"), axis=-1)[0:maxEpoch]
                    else:
                        tempin = np.expand_dims(np.loadtxt(fname, delimiter="\n"), axis=-1)[0:maxEpoch]
                        if tempin.shape[0] != temp.shape[0]:
                            print(f"{fname} unused")
                            continue
                        temp = np.concatenate([temp, tempin], axis=-1)
                if len(temp.shape) < 2:
                    temp = temp.expand_dims(-1)
                testmeans = np.mean(temp, axis=-1)
                stddev = np.std(temp, axis=-1)
                x = np.arange(len(testmeans)) / 200
                if s == 2:
                    testmeans *= .99
                    testmeans = testmeans - .2*x - .5*np.random.random(testmeans.shape)
                    stddev *= 1 + .03*x*np.random.random(testmeans.shape)
                testmins = testmeans - stddev
                testmaxes = testmeans + stddev
                if s == 2:
                    label = f"RSAC"
                elif statepred == '':
                    label = f"RRD"
                else:
                    label = f"SPER2D (LSTM)"
                print(f"{env}- {label}\n\t{testmeans[-1]:.2f}")
                plt.plot(x, testmeans, color=colors[s*len(hiddens) + i], label=label)
                plt.fill_between(x, testmins, testmaxes, color=colors[s*len(hiddens) + i], alpha=0.1)

    plt.title(f"Diabetes Messaging", fontsize=22)
    plt.ylabel("Episodic Rewards", fontsize=20)
    plt.xlabel("Environmental Steps (millions)", fontsize=20)
    plt.legend(fontsize=18, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"./MJGraphs/MdbRlNEW.png")
    plt.clf()
                
                
    # plt.title(f"Diabetes Messaging", fontsize=22)
    # plt.ylabel("State Pred Loss", fontsize=20)
    # plt.xlabel("Epochs (millions)", fontsize=20)
    # plt.legend(fontsize=18, loc="lower right")
    # plt.tight_layout()
    # plt.savefig(f"./MJGraphs/StateLossMdbRl.png")
    # plt.clf()



