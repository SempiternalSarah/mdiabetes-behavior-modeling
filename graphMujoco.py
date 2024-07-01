import numpy as np
import matplotlib.pyplot as plt
import os

# envs = ["Ant-v2", "HalfCheetah-v2", "Humanoid-v2"]
envs = ["Ant-v2"]
# envs = ["Humanoid-v2"]
seedmin, seedmax = 1, 5
hiddens = [1, 2, 3]
# hiddens = [1]
# htimes = [5]
htimes = [1]
# qlrs = [0.0003, 0.0006, 0.0001]
# alrs = [0.0003, 0.0006, 0.0001]
qlrs = [0.0003]
alrs = [0.0003]
qlr = 0.0003
alr = 0.0003
starts = [10000]
start2 = 1000000
# statepred = "nn"
statepreds = ['lstm']
hsizes=[128]
statelrs = [0.001]
slr = 0.001
# statelrs = [0.0005]
contexts = [20]
variables = ['rewards']
variables = ['statepredloss', 'statepredtestloss']
# variables = ['statepredloss']
# variables = ['actloss']
# variables = ['qfloss']
variables=['rrdloss']

maxEpoch = 350

colors = plt.cm.rainbow(np.linspace(0, 1, 5))
for env in envs:
    for i, hidden in enumerate(hiddens):
        curCol = 0
        for v, variable in enumerate(variables):
            for context in contexts:
                for hs in hsizes:
                    for start in starts:
                        for s, statepred in enumerate(statepreds):
                            for sacount, htime in enumerate(htimes):
                                if statepred == '':
                                    folder = f"./saved_mujoco/{env}/NoContext/{variable}"
                                else:
                                    if htime != 1:
                                        continue
                                    folder = f"./saved_mujoco/{env}/NoContext/{statepred}/{variable}"
                                temp = None
                                for seed in range(seedmin, seedmax+1):
                                    #{args.numHidden}Hidden{args.seed}QLR{args.qlr}ALR{args.actlr}SLR{args.statelr}Start{args.startLearning}HS{args.hiddenSizeLSTM}C{args.context}.csv
                                    fname = f"{folder}/{hidden}Hidden{seed}QLR{qlr}ALR{alr}SLR{slr}Start{start},{start2}HS{hs}C{context}.csv"

                                    if not os.path.isfile(fname):
                                        print(f"NOT FOUND: {fname}")
                                        continue
                                    tempin = np.expand_dims(np.loadtxt(fname, delimiter="\n"), axis=-1)[0:maxEpoch]
                                    if tempin.shape[0] < maxEpoch:
                                        print(f"{fname} unused")
                                        continue
                                    if temp is not None:
                                        temp = np.concatenate([temp, tempin], axis=-1)
                                    else:
                                        temp = tempin
                                if len(temp.shape) < 2:
                                    temp = temp.expand_dims(-1)
                                if 'state' in variable:
                                    # temp = np.log(temp)
                                    None
                                testmeans = np.mean(temp, axis=-1)
                                stddev = np.std(temp, axis=-1)
                                x = np.arange(len(testmeans)) / 200
                                label = f"{variable}"
                                testmins = testmeans - stddev
                                testmaxes = testmeans + stddev
                                plt.plot(x, testmeans, color=colors[curCol], label=label)
                                print(f"{env}({hidden})-{label}\n\t{testmeans[-1]:.2f}")
                                plt.fill_between(x, testmins, testmaxes, color=colors[curCol], alpha=0.1)
                                curCol += 1

        plt.title(f"{env} (3 Hidden)", fontsize=22)
        if 'qfloss' in variables:
            plt.ylabel("Loss (MSE)", fontsize=20)
            name = "QFLOSS"
        elif 'statepredloss' in variables:
            plt.ylabel("Loss (MSE)", fontsize=20)
            name = "ASTATEPRED"
        elif 'reward' in variables:
            plt.ylabel("Episodic Returns", fontsize=20)
            name = "AREWARD"
        else:
            plt.ylabel("Loss", fontsize=20)
            name = variables[0]
        plt.xlabel("Epochs (millions)", fontsize=20)
        handles, labels = plt.gca().get_legend_handles_labels()
        order = np.arange(len(handles))
        # order = [0,1,4,3,2]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=12, loc="lower left")
        plt.tight_layout()
        print(f"./MJGraphs/{name}{env}({hidden}).png")
        plt.savefig(f"./MJGraphs/{name}{env}({hidden}).png")
        plt.clf()



