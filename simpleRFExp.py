from experiment import Experiment
from utils.behavior_data import BehaviorData
from visuals import Plotter
import torch
import numpy as np
from utils.state_data import StateData
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

bd = BehaviorData(minw=2, maxw=31, include_state=True, full_questionnaire=True, include_pid=False, expanded_states=True, top_respond_perc=.5, split_weekly_questions=True)

trainLabels = bd.labels[bd.train].numpy()
trainFeatures = bd.features[bd.train].numpy()
# trainFeatures = trainFeatures[trainLabels[:, 0] != 1]
# trainFeatures = trainFeatures[:, 1:]
# trainLabels = trainLabels[trainLabels[:, 0] != 1]
# trainLabels = trainLabels[:, 1:]

testLabels = bd.labels[bd.test].numpy()
testFeatures = bd.features[bd.test].numpy()
# testFeatures = testFeatures[testLabels[:, 0] != 1]
# testFeatures = testFeatures[:, 1:]
# testLabels = testLabels[testLabels[:, 0] != 1]
# testLabels = testLabels[:, 1:]

rf = RandomForestClassifier(n_estimators=10, max_depth=10, min_samples_split = 2, bootstrap=True)
rf.fit(trainFeatures, trainLabels)
preds = rf.predict(trainFeatures)
# print(preds)
accuracy = (trainLabels * preds).sum() / (2 * preds.shape[0])
print(accuracy)

preds = rf.predict(testFeatures)
print(preds.sum(axis=0))
accuracy = (testLabels * preds).sum() / (2 * preds.shape[0])
print(accuracy)