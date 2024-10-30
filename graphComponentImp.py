import numpy as np
import matplotlib.pyplot as plt
import os

# envs = ["Humanoid-v2"]
env = "Diabetes"
seedmin, seedmax = 1, 40
hiddens = [1]
statepreds = ['', 'lstm']
variable = 'partimp'
maxEpoch = 200
qlr = 3e-5
hsize = 128
eqs = ["True"]
elem_map = [
      "Fruit Consumption",
      "Veg. Consumption Avg.",
      "Healthy Snacks and Breakfast",
      "Other Healthy Food",
      "Tobacco Avoidance",
      "Alcohol Avoidance",
      "Unhealthy Restaurants",
      "Unhealthy Junk Food",
      "Fitness Overall",
      "Sports Activity",
      "Gym Workouts",
      "Incidental Exercise",
      "Diabetes Cause Knowledge",
      "Diabetes Cause Knowledge (Weight)",
      "Diabetes Cause Knowledge (Diet)",
      "Diabetes Cause Knowledge (Exercise)",
      "Diabetes Complication Knowledge",
]
elem_map_fname = []
for k in elem_map:
    temp = k.replace("/", "-")
    temp = temp.replace(" ", "_")
    elem_map_fname.append(temp) 


np.random.seed(0)

colors = plt.cm.rainbow(np.linspace(0, 1, len(statepreds)))
for eq in eqs:
    testmeans = {}
    testmaxes = {}
    testmins = {}

    for s, statepred in enumerate(statepreds):
        temp = None
        folder = "./saved_mdiabetes_rl"
        if statepred == '':
            folder = f"{folder}/trajlog"
        else:
            folder = f"{folder}/{statepred}/trajlog"
        temps = []
        for seed in range(seedmin, seedmax+1):
            fname = f"{folder}/{variable}{seed}H{hsize}LR{qlr}EQ{eq}.csv"
            if not os.path.isfile(fname):
                print(f"NOT FOUND: {fname}")
                continue
            with open(fname) as f:
                temp = np.loadtxt((x.replace(', ', ' ') for x in f))
                if len(temp) > 200:
                    temp = temp[199:199+maxEpoch]
                else:
                    temp = temp[:maxEpoch]
                temps.append(temp)
        #         print(temp.shape)
        # exit()
        alls = np.stack(temps, axis=-1)
        # print(alls.shape)
        testmeans[statepred] = np.mean(alls, axis=-1)
        stddev = np.std(alls, axis=-1)
        
        testmins[statepred] = testmeans[statepred] - stddev
        testmaxes[statepred] = testmeans[statepred] + stddev



    for i in range(17):
        for s, statepred in enumerate(statepreds):
            if statepred == '':
                label = f"RRD"
            else:
                label = f"SPER2D (LSTM)"
            print(f"{elem_map[i]}- {label}\n\t{(testmeans[statepred][:, i][-1] * 100):.2f} - {(testmaxes[statepred][:, i][-1] - testmeans[statepred][:, i][-1]) * 100/np.sqrt(40):.2f}")
            # print(testmeans[statepred].shape)
            x = np.arange(testmeans[statepred].shape[0]) / 200
            plt.plot(x, testmeans[statepred][:, i], color=colors[s], label=label)
            plt.fill_between(x, testmins[statepred][:, i], testmaxes[statepred][:, i], color=colors[s], alpha=0.1)

        plt.title(f"{elem_map[i]}", fontsize=22)
        plt.ylabel("% of Participants Improved", fontsize=20)
        plt.xlabel("Environmental Steps (millions)", fontsize=20)
        plt.legend(fontsize=18, loc="lower right")
        plt.tight_layout()
        print(f"./MJGraphs/{variable}{elem_map_fname[i]}EQ{eq}form.png")
        plt.savefig(f"./MJGraphs/{variable}{elem_map_fname[i]}EQ{eq}form.png")
        plt.clf()
                    
                    



