import numpy as np
import matplotlib.pyplot as plt
import os

envs = ["Ant-v2", "HalfCheetah-v2"]
# envs = ["Hopper-v2", "Walker2d-v2"]

# envs = ["Ant-v2"]
# envs = ["Humanoid-v2"]
# envs = ["HalfCheetah-v2"]
seedmin, seedmax = 1, 5
hiddens = [1, 2, 3]
# hiddens = [0]
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
statepreds = ['lstm', '', 'odt']
# statepreds = ['lstm']
# statepreds = ['']
hsizes=[128]
statelrs = [0.001]
slr = 0.001
# statelrs = [0.0005]
contexts = [20]
variables = ['rewards']
# variables = ['statepredloss', 'statepredtestloss']
# variables = ['statepredloss']
# variables = ['actloss']
# variables = ['qfloss']
# variables=['rrdloss']

colors = plt.cm.rainbow(np.linspace(0, 1, 5))
for maxEpoch in range(400, 600, 25):
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
                                        folder = f"./saved_mujoco/{env}/NoContext/Fill/{variable}"
                                    elif statepred =='odt':
                                        folder = f"../CONTAINMENTZONE/saved_mujoco/{env}/{variable}"
                                    else:
                                        if htime != 1:
                                            continue
                                        folder = f"./saved_mujoco/{env}/NoContext/{statepred}/{variable}"
                                    temp = None
                                    for seed in range(seedmin, seedmax+1):
                                        #{args.numHidden}Hidden{args.seed}QLR{args.qlr}ALR{args.actlr}SLR{args.statelr}Start{args.startLearning}HS{args.hiddenSizeLSTM}C{args.context}.csv
                                        if statepred == 'lstm':
                                            fname = f"{folder}/{hidden}Hidden{seed}QLR{qlr}ALR{alr}SLR{slr}Start{start},{start2}HS{hs}C{context}.csv"
                                        elif statepred == 'odt':
                                            fname = f"{folder}/{hidden}Hidden10HTime{seed}LR0.0003.csv"
                                        else:
                                            fname = f"{folder}/{hidden}Hidden{seed}QLR{qlr}ALR{alr}SLR{slr}Start{start},{start2}HS128C{context}.csv"

                                        if not os.path.isfile(fname):
                                            print(f"NOT FOUND: {fname}")
                                            continue
                                        # print(tempin.shape)
                                        if statepred != 'odt':
                                            tempin = np.expand_dims(np.loadtxt(fname, delimiter="\n"), axis=-1)[start2//5000:maxEpoch]
                                            if tempin.shape[0] < maxEpoch - start2/5000:
                                                # print(f"{fname} unused")
                                                continue
                                        else:
                                            tempin = np.expand_dims(np.loadtxt(fname, delimiter="\n"), axis=-1)[:maxEpoch - start2//5000]
                                        # print(tempin.shape)
                                        if temp is not None:
                                            temp = np.concatenate([temp, tempin], axis=-1)
                                        else:
                                            temp = tempin
                                    if temp is None:
                                        # print("NO SEEDS USED!!")
                                        continue
                                    if len(temp.shape) < 2:
                                        temp = temp.expand_dims(-1)
                                    if 'state' in variable:
                                        # temp = np.log(temp)
                                        None
                                    if temp.shape[-1] < 2:
                                        continue
                                    # temp += temp * .0001*(np.random.randn(*temp.shape))
                                    testmeans = np.mean(temp, axis=-1)
                                    stddev = np.std(temp, axis=-1)
                                    x = np.arange(len(testmeans)) / 200
                                    if statepred == 'lstm':
                                        label = "SPER2D"
                                    elif statepred == '':
                                        label = "RRD"
                                    else:
                                        label = "ODT"
                                    testmins = testmeans - stddev
                                    testmaxes = testmeans + stddev
                                    plt.plot(x, testmeans, color=colors[curCol], label=label)
                                    print(f"{env}({hidden})-{label}\n\t{testmeans[-1]:.2f}±{stddev[-1]/np.sqrt(10):.2f}")
                                    plt.fill_between(x, testmins, testmaxes, color=colors[curCol], alpha=0.1)
                                    curCol += 1

            plt.title(f"{env} ({hidden} Hidden)", fontsize=22)
            if 'qfloss' in variables:
                plt.ylabel("Loss (MSE)", fontsize=20)
                name = "QFLOSS"
            elif 'statepredloss' in variables:
                plt.ylabel("Loss (MSE)", fontsize=20)
                name = "ASTATEPRED"
            elif 'rewards' in variables:
                plt.ylabel("Episodic Returns", fontsize=20)
                name = "rewards"
            else:
                plt.ylabel("Loss", fontsize=20)
                name = variables[0]
            plt.xlabel("Timesteps (millions)", fontsize=20)
            handles, labels = plt.gca().get_legend_handles_labels()
            if len(handles) < len(statepreds):
                plt.clf()
                continue
            order = np.arange(len(handles))
            # order = [0,1,4,3,2]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=18, loc="lower left")
            plt.tight_layout()
            print(f"./MJGraphs/{name}{statepred}{env}({hidden}).png")
            plt.savefig(f"./MJGraphs/{name}{statepred}{env}({hidden}).png")
            plt.clf()



