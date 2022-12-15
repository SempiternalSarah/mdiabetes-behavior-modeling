import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from utils.state_data import StateData
from utils.content import StatesHandler, QuestionHandler, MessageHandler
from torch import save, load
import torch

# class to manage loading and encoding behavioral data
class BehaviorData:
    
    def __init__(self, 
                 minw=2, maxw=31, 
                 include_pid=False, include_state=True, 
                 active_samp=None, 
                 window=3,
                 train_perc=.8,
                 expanded_states=True,
                 top_respond_perc=1.0,
                 full_questionnaire=False,
                 full_sequence=False,
                 insert_predictions=False):
        # minw, maxw: min and max weeks to collect behavior from
        # include_pid: should the participant id be a feature to the model
        # include_state: should the participant state be a feature

        # whether to include the full sequence of weekly message/question/response
        # used for simple (non LSTM) models
        self.full_sequence = full_sequence
        # whether to use feature modifications to replace non responses with predicted responses
        self.insert_predictions = insert_predictions
        # whether to use each question in the questionnaire as features
        self.full_questionnaire = full_questionnaire
        # what percent of participants to use (taken from the top responders)
        self.top_respond_perc = top_respond_perc
        # min and max weeks to load data from 
        # (2 and 31 for the whole dataset, other values probably break things now)
        self.minw, self.maxw = minw, maxw
        # whether to use participant ID as a feature
        # not used
        self.include_pid = include_pid
        # whether to use new (expanded) state mappings
        # default to true
        self.expanded_states = expanded_states
        # whether to include state values as features at all
        # default to true
        self.include_state = include_state
        self.active_samp = active_samp if active_samp is not None else 1
        # not used
        self.window = window
        # calculate file name for storing/loading data
        self.fname = f"{self.minw}-{self.maxw}{self.include_pid}{self.include_state}{self.expanded_states}{self.full_questionnaire}{self.full_sequence}{self.top_respond_perc}.pickle"

        # data saved - we can just load it
        if os.path.exists(self.fname):
            self.load()
        else:
            self.data = self.build()
            self.features, self.labels, self.featureList = self.encode(self.data)
            self.save()
        # find index of first response value so we don't have to compute this again
        # used to replace non responses with the model's prediction
        for idx, feature in enumerate(self.featureList):
            if (feature == "response0_1"):
                self.responseIdx = idx

        self.splitData(train_perc)

        # set up our response modifications
        # these will be used to replace non responses with the model's prediction
        self.responseMods = {}
        for idx in self.train:
            self.responseMods[idx] = np.zeros_like(self.chunkedFeatures[idx])
        for idx in self.test:
            self.responseMods[idx] = np.zeros_like(self.chunkedFeatures[idx])

        
    # splits data into test and training
    def splitData(self, train_perc):
        numParticipants = len(self.nzindices)
        numTrainParticipants = int(train_perc * numParticipants)
        self.train = np.random.choice(numParticipants, numTrainParticipants)
        self.test = [idx for idx in range(numParticipants) if idx not in self.train]
    
        self.chunkedFeatures = torch.tensor_split(self.features, self.nzindices)
        self.chunkedLabels = torch.tensor_split(self.labels, self.nzindices)


    # get features for a participant
    def get_features(self, idx):
        if (self.full_sequence and self.insert_predictions):
            return self.chunkedFeatures[idx] + self.responseMods[idx]
        else:
            return self.chunkedFeatures[idx]

    # set feature modifications for all participants
    def set_feature_response_mods(self, indx, preds):
        # do nothing if we're not using responses as features
        # modifications will remain 0
        if (not self.full_sequence or not self.insert_predictions):
            return
        # set up our feature modifications
        mods = np.zeros_like(self.chunkedFeatures[indx].numpy())
        # iterate through each week (row of this participants data)
        for i, weekRow in enumerate(self.chunkedFeatures[indx]):
            # iterate through the responses of weeks before this one
            for j in range(i):
                # get index of this week's response to q1
                idx = self.responseIdx + (2 * j)
                for offset in range(2):
                    if weekRow[idx + offset] == -1:
                        # first question
                        if (offset == 0):
                            # calculate most likely predicted class and save to use as the feature
                            mods[i][idx + offset] = 2 + np.argmax(preds[j][0:(self.dimensions[1]//2)])
                            # need to add 2 (feature itself is -1, argmax is 0 if pred class is 1)
                        else:
                            mods[i][idx + offset] = 2 + np.argmax(preds[j][(self.dimensions[1]//2):])
        self.responseMods[indx] = mods
                    


    # load state information from the baseline questionnaire
    def load_questionnaire_states(self):
        if (self.full_questionnaire):
            sh = StatesHandler(map="map_individual.json")
        elif (self.expanded_states):
            sh = StatesHandler(map="map_detailed.json")
        else:
            sh = StatesHandler(map="map.json")
        whatsapps, states = sh.compute_states()
        def modify_whatsapp(x):
            # helper function to parse the whatsapp numbers
            x = str(x)
            x = x[len(x)-10:]
            return int(x)
        participantIDs = torch.tensor(np.loadtxt("arogya_content/all_ai_participants.csv", delimiter=",", skiprows=1, dtype="int64"))
        participantIDs[:, 1].apply_(modify_whatsapp)
        start_weeks = self.get_participant_start_weeks()
        
        # filter responses to only include ones in the AI participant set
        isect, idIdxs, stateIdxs = np.intersect1d(participantIDs[:, 1], whatsapps, return_indices=True)
        # combine the glific IDs with the states into a dictionary and return
        return dict(zip(participantIDs[idIdxs, 0].numpy(), states[stateIdxs].numpy())), start_weeks

    def get_participant_start_weeks(self):
        # get start week for each participant
        startWeeks = {}
        for week in range(2, 32):
            pids = np.loadtxt(f"./local_storage/prod/responses/participant_responses_week_{week}.csv", delimiter=",", skiprows=1, dtype="str")
            pids = pids[:, 0].astype("int64")
            for pid in pids:
                if pid not in startWeeks:
                    startWeeks[pid] = week
        return startWeeks

    
    def filter_top_responders(self, df: pd.DataFrame):
        # get WEEKLY response count for each participant
        df["response_count"] = df.apply(lambda row: np.count_nonzero(row["response"]), axis=1)
        # sum response count for each participant
        counts = df.groupby("pid")['response_count'].sum().sort_values()
        # find cutoff value (participants with fewer total responses are removed)
        cutoff_idx = int((len(counts)) * (1 - self.top_respond_perc))
        print(cutoff_idx, len(counts))
        # select top responders
        df = df[df["pid"].isin(counts[cutoff_idx:].keys())]
        # save response counts for later use
        self.counts = counts

        return df


    def build(self):
        # call StateData and build our initial unencoded dataset
        sd = StateData(detailed=self.expanded_states)
        d = sd.build(minw=self.minw, maxw=self.maxw)
        enc = OrdinalEncoder().fit_transform
        # load dictionary of pids to states
        init_states, start_weeks = self.load_questionnaire_states()
        # filter out rows that aren't supposed to be here
        # unsure where they come from but they aren't listed in the all_ai_participants.csv
        d = d[d["pid"].isin(list(dict.keys(init_states)))]
        # change the computed states to be the initial questionnaire states instead
        d["state"] = d.apply(lambda row: init_states[row["pid"]], axis=1)

        # adjust week values per participant (their first week should be 0, last 24)
        def adjustWeek(row):
            w = row["week"] - start_weeks[row["pid"]]
            return w
        d["week"] = d.apply(adjustWeek, axis=1)

        # participants are in for 24 weeks (??)
        # first response from final group of participants seen week 8
        # last responses received week 31
        # THIS SEEMS CORRECT FROM THE DATA BUT POSSIBLE IM MISSING SOMETHING

        # filter out rows after participant completed study
        d = d[d["week"] < 24]

        # rescale pid values in case we want to use them as features
        d["pidFeat"] = enc(d["pid"].values.reshape(-1,1)).astype(int)
        # sort by week
        d = d.sort_values(by="week")
        # sort by participant, keeping the week sort
        # result is sorted by participants
        # and participant rows are sorted in increasing week order
        d = d.sort_values(by="pidFeat", kind="stable")

        # select top responders based on parameter passed in constructor
        d = self.filter_top_responders(d)
        # check if we need to build up the full sequence
        if (self.full_sequence):
            # method to insert a week's element into all other rows
            # TODO: FIGURE OUT HOW TO WRITE THIS MORE EFFICIENTLY
            def construct_week_elem(row, weekno, elem):
                if ((weekno < row['week']) or (elem != "response" and weekno == row['week'])):
                    # use full knowledge of the past
                    temp = d[d["pid"] == row['pid']]
                    temp = temp[temp["week"] == weekno]
                    # set non response to -1 to distinguish from unknown (future) response
                    if (elem == "response" and 0 in temp.iloc[0][elem]):
                        vals = []
                        for val in temp.iloc[0][elem]:
                            if (val == 0):
                                vals.append(-1)
                            else:
                                vals.append(val)
                        return tuple(vals)
                    else:
                        return tuple(temp.iloc[0][elem])
                else:
                    # don't use knowledge of the future
                    return (0, 0)
            # go through and insert msg/question/response for all weeks as appropriate
            for elem in ["paction_sids", "pmsg_ids", "qids", "response"]:
                print(elem)
                for week in range(24):
                    d[f"{elem}{week}"] = d.apply(lambda row: construct_week_elem(row, week, elem), axis=1, result_type='reduce')

        # record splits between different participants for later
        # basically we want to easily extract individual participant data
        # indices will REMAIN THE SAME for the encoded pytorch tensor
        # nonzero() returns a 1 element tuple, unpack, take 0th entry off (it's 0)
        # convert to list for later use
        self.nzindices = d["pid"].diff().to_numpy().nonzero()[0][1:].tolist()
        return d
    
    def encode(self, data: pd.DataFrame):
        # encode the row locations of data
        # data: pd.DataFrame
        X, Y = [], []
        for idx, row in data.iterrows():
            x, y, featureList = self.encode_row(row)
            X.append(x)
            Y.append(y)
        X, Y = np.stack(X), np.stack(Y)
        print(X.shape, Y.shape, len(featureList))
        return torch.tensor(X).float(), torch.tensor(Y).float(), np.array(featureList)
                
    def encode_row(self, row):
        # here we take a row from the main behavior dataset and 
        # encode all of the features for our model
        # Features:
        #  - participant ID                    (enumeration of participants) (NOT USED NORMALLY)
        #  - dynamic state elements            (real values between (1,3)
        #  - action id which prompted question (enumerated 1-5,  binary encoded)
        #  - message ids of action             (enumerated 1-57, binary encoded)
        #  - question ids to predict repsonse  (enumerated 1-32, binary encoded
        def _padded_binary(a, b):
            # helper function to binary encode a and 
            # pad it to be the length of encoded b
            a, b = int(a), int(b)
            l = len(format(b,"b"))
            a = format(a,f"0{l}b")
            return np.array([int(_) for _ in a])
        def _onehot(a, l):
            vec = np.zeros(l)
            vec[a] = 1
            return vec

        featureList = []

        elems = []
        # max value for each (state elem, message id, question id) for padding
        ls = []
        if (self.expanded_states):
            maxSVal = 17
        else:
            maxSVal = 5
        if (self.full_sequence):
            for week in range(24):
                for elem in ["paction_sids", "pmsg_ids", "qids"]:
                    elems.append(f"{elem}{week}")
                ls += [maxSVal,57,32]
        else:
            elems = ["paction_sids", "pmsg_ids", "qids"]
            ls = [maxSVal,57,32]
        feats_to_enc = np.array(row[elems].values)
        feats_to_enc = feats_to_enc.tolist()
        
        if self.include_pid:
            X = np.array([row["pidFeat"]])
            featureList.append("pidFeat")
        else:
            X = np.array([])
        if self.include_state:
            X = np.append(X, row["state"])
            featureList += ["state"] * len(row["state"])

        if self.full_sequence:
            for week in range(24):
                X = np.append(X, row[f"response{week}"])
                featureList += [f"response{week}_1", f"response{week}_2"]
        
        for j in range(len(feats_to_enc)):
            for k in range(len(feats_to_enc[j])):
                # encode the feature and add it to our feat vector
                bin_feat = _padded_binary(feats_to_enc[j][k],ls[j])
                X = np.append(X, bin_feat)
                featureList += [f"{elems[j]}_{k}"] * len(bin_feat)
        # responses are the labels
        Y = np.array([])
        for i,r in enumerate(row["response"]):
            Y = np.append(Y, _onehot(r,4))
        return X, Y, featureList
    
    def save(self):
        out = {"data": self.data, "features": self.features, "labels":self.labels, "featureList": self.featureList, "nzIndices": self.nzindices}
        save(out, self.fname)
        
    def load(self):
        d = load(self.fname)
        self.features = d["features"]
        self.data = d["data"]
        self.labels = d["labels"]
        self.featureList = d["featureList"]
        self.nzindices = d["nzIndices"]
        
    @property
    def dimensions(self):
        # helper to get the x and y input dimensions
        return self.features.shape[1], self.labels.shape[1] - 2
        
