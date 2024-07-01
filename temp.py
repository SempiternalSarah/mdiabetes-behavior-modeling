from experiment import Experiment
from utils.behavior_data import BehaviorData
from utils.content import StatesHandler
from visuals import Plotter
import torch
import numpy as np
from utils.state_data import StateData
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

elem_map = [
      "Fruit Consumption",
      "Veg. Consumption Avg.",
      "Veg. Consumption Recent",
      "High Fat Food Avoidance",
      "Sports Activity",
      "Formal Workouts",
      "Walking",
      "Sports/Workout/Walking",
      "Daily Avg. Exercise Time",
      "Incidental Exercise",
      "Cause Knowledge",
      "Complication Knowledge"
]


ageMatching = [0, 1, 2, 3]
# ageMatching = [2, 3]
# ageMatching = [0, 1]

# genderMatching = [1]
genderMatching = [1, 3]
# genderMatching = [3]

incomeMatching = [1, 3]
# incomeMatching = [1]
# incomeMatching = [3]

educationMatching = [0, 1, 2, 3]
# educationMatching = [0]
# educationMatching = [1, 2, 3]


respondPerc = 1.0


def load_questionnaire_states(endline=False, detail=0, aiset=True):
        if (detail > 2):
            sh = StatesHandler(map="map_questionnaire_final.json", endline=endline)
        elif (detail > 1):
            sh = StatesHandler(map="map_individual.json", endline=endline)
        elif (detail > 0):
            sh = StatesHandler(map="map_detailed.json", endline=endline)
        elif (detail > -1):
            sh = StatesHandler(map="map.json", endline=endline)
        else:
            sh = StatesHandler(map="map_traditional.json", endline=endline)
        whatsapps, states, slist = sh.compute_states()
        def modify_whatsapp(x):
            # helper function to parse the whatsapp numbers
            x = str(x)
            x = x[len(x)-10:]
            return int(x)
        participantIDs = torch.tensor(np.loadtxt("arogya_content/all_ai_participants.csv", delimiter=",", skiprows=1, dtype="int64"))
        participantIDs[:, 1].apply_(modify_whatsapp)
        
        # filter responses to only include ones in the AI participant set
        isect, idIdxs, stateIdxs = np.intersect1d(participantIDs[:, 1], whatsapps, return_indices=True)
        if (aiset):
            # combine the glific IDs with the states into a dictionary and return
            return dict(zip(participantIDs[idIdxs, 0].numpy(), states[stateIdxs].numpy()))
        else:
            stats = np.delete(states.numpy(), stateIdxs, axis=0)
            wapps = np.delete(whatsapps.numpy(), stateIdxs)
            return dict(zip(wapps, stats))
        
post = load_questionnaire_states(True, 3, True)
print(len(post))
pre = load_questionnaire_states(False, 3, True)
print(len(pre))


diffs = []
demos = []
AIids = []

bd = BehaviorData(minw=2, maxw=29, include_state=True, include_pid=False, top_respond_perc=respondPerc)
print(bd.data)
for glifid in post.keys():
    if glifid in pre and glifid in bd.data['pid'].to_numpy():
        diffs.append(post[glifid] - pre[glifid])
        AIids.append(glifid)
        demos.append(pre[glifid][-5:])
    else:
        None
        #print(glifid)

print(AIids)
print(len(diffs))
np.savetxt("GlifIDsAIAll.csv", AIids, delimiter='\n', fmt="%i")

demos = np.array(demos)
allDiffValsAI = np.array(diffs)[:, 0:-5]