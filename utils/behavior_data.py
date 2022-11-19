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
                 minw=2, maxw=8, 
                 include_pid=True, include_state=True, 
                 active_samp=None, 
                 window=3,
                 load=None,
                 train_perc=.8,
                 expanded_states=False,
                 top_respond_perc=1.0,
                 full_questionnaire=False):
        # minw, maxw: min and max weeks to collect behavior from
        # include_pid: should the participant id be a feature to the model
        # include_state: should the participant state be a feature
        if load is not None:
            self.load(load)
            return
        self.full_questionnaire = full_questionnaire
        self.top_respond_perc = top_respond_perc
        self.minw, self.maxw = minw, maxw
        self.include_pid = include_pid
        self.expanded_states = expanded_states
        self.include_state = include_state
        self.active_samp = active_samp if active_samp is not None else 1
        self.window = window
        self.data = self.build()
        self.features, self.labels = self.encode(self.data)
        self.splitData(train_perc)
        
    # splits data into test and training
    def splitData(self, train_perc):
        numParticipants = len(self.nzindices)
        numTrainParticipants = int(train_perc * numParticipants)
        self.train = np.random.choice(numParticipants, numTrainParticipants)
        self.test = [idx for idx in range(numParticipants) if idx not in self.train]
    
        self.chunkedFeatures = torch.tensor_split(self.features, self.nzindices)
        self.chunkedLabels = torch.tensor_split(self.labels, self.nzindices)

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
        
        # filter responses to only include ones in the AI participant set
        isect, idIdxs, stateIdxs = np.intersect1d(participantIDs[:, 1], whatsapps, return_indices=True)
        # combine the glific IDs with the states into a dictionary and return
        return dict(zip(participantIDs[idIdxs, 0].numpy(), states[stateIdxs].numpy()))

    
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
        init_states = self.load_questionnaire_states()
        # filter out rows that aren't supposed to be here
        # unsure where they come from but they aren't listed in the all_ai_participants.csv
        d = d[d["pid"].isin(list(dict.keys(init_states)))]
        # change the computed states to be the initial questionnaire states instead
        # ideally we'd avoid loading the state data that we aren't going to use but
        # reworking that data flow is a lot harder than leaving it intact and patching here
        d["state"] = d.apply(lambda row: init_states[row["pid"]], axis=1)

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

        # record splits between different participants for later
        # basically we want to easily extract individual participant data
        # indices will REMAIN THE SAME for the encoded pytorch tensor
        # nonzero() returns a 1 element tuple, unpack, take 0th entry off (it's 0)
        # convert to list for later use
        self.nzindices = d["pid"].diff().to_numpy().nonzero()[0][1:].tolist()
        return d
    
    def encode(self, data):
        # encode the row locations of data
        # data: pd.DataFrame
        X, Y = [], []
        for idx, row in data.iterrows():
            x, y = self.encode_row(row)
            X.append(x)
            Y.append(y)
        X, Y = np.stack(X), np.stack(Y)
        print(X.shape, Y.shape)
        return torch.tensor(X).float(), torch.tensor(Y).float()
                
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
        feats_to_enc = np.array(row[["paction_sids", "pmsg_ids", "qids"]].values)
        feats_to_enc = feats_to_enc.tolist()
        
        if self.include_pid:
            X = np.array([row["pidFeat"]])
        else:
            X = np.array([])
        if self.include_state:
            X = np.append(X, row["state"])
        # max value for each (state elem, message id, question id) for padding
        if (self.expanded_states):
            maxSVal = 17
        else:
            maxSVal = 5
        ls = [maxSVal,57,32]
        for j in range(len(feats_to_enc)):
            for k in range(len(feats_to_enc[j])):
                # encode the feature and add it to our feat vector
                bin_feat = _padded_binary(feats_to_enc[j][k],ls[j])
                X = np.append(X, bin_feat)
        # responses are the labels
        Y = np.array([])
        for i,r in enumerate(row["response"]):
            Y = np.append(Y, _onehot(r,4))
        return X, Y
    
    def save(self, p):
        out = {"minw": self.minw, "maxw": self.maxw, "include_pid": self.include_pid,
               "include_state": self.include_state, "active_samp": self.active_samp,
               "window": self.window, "data": self.data}
        save(out, p)
        
    def load(self, p):
        d = load(p)
        self.minw, self.maxw = d["minw"], d["maxw"]
        self.include_pid = d["include_pid"]
        self.include_state = d["include_state"]
        self.active_samp = d["active_samp"]
        self.window = d["window"]
        self.data = d["data"]
        
    @property
    def dimensions(self):
        # helper to get the x and y input dimensions
        return self.features.shape[1], self.labels.shape[1] - 1
        
