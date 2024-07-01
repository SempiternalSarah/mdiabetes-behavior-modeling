import gym
import argparse
import random
import gym.spaces
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Independent
from models.decision_transformer import DecisionTransformer
import os
import time


def toBool(x):
    return (str(x).lower() in ['true', '1', 't'])

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--actlr", type=float, default=3e-4, help="Actor learning rate")
parser.add_argument("--statelr", type=float, default=1e-3, help="State pred learning rate")
parser.add_argument("--context", type=int, default=20, help="No. timesteps history for state prediction")
parser.add_argument("--qlr", type=float, default=3e-4, help="critic learning rate")
parser.add_argument("--alpha_lr", type=float, default=3e-4, help="alpha learning rate")
parser.add_argument("--alpha", type=float, default=0.1, help="Entropy/regularization")
parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
parser.add_argument("--numSteps", type=int, default=40000, help="iterations of overall algorithm")
parser.add_argument("--startLearning", type=int, default=1000000, help="no. samples taken before RRD and SAC training occurs")
parser.add_argument("--startLearningState", type=int, default=10000, help="no. samples taken before state pre-training occurs")
parser.add_argument("--bufferSize", type=int, default=1000000, help="no. samples in buffer")
parser.add_argument("--numHidden", type=int, default=0, help="no. parts of robot invisible each time step")
parser.add_argument("--train_batches", type=int, default=100, help="no. batches per timestep")
parser.add_argument("--envSteps", type=int, default=100, help="no. environment steps per timestep")
parser.add_argument("--consecHidden", type=int, default=1, help="no. consecutive timesteps to hide information (e.g. hide an ant limb for 10 environmental steps)")
parser.add_argument("--env", type=str, default="HalfCheetah-v2", help="MuJoCo environment")
parser.add_argument("--logging", type=toBool, default=False)
parser.add_argument("--cuda", type=toBool, default=False)
parser.add_argument("--statepred", type=toBool, default=False)
parser.add_argument("--statefill", type=toBool, default=True)
parser.add_argument("--contextSAC", type=toBool, default=False)
parser.add_argument("--statemodel", type=str, default="lstm")
parser.add_argument("--numBreaks", type=int, default=4, help="Minibatches for state pred training")
parser.add_argument("--stateTrainMult", type=int, default=1, help="Multiplier for state learning epochs")
parser.add_argument("--hiddenSizeLSTM", type=int, default=64)



args = parser.parse_args()

# overwrite
if args.cuda:
    args.cuda = torch.cuda.is_available()
    print(f"Using Cuda? {args.cuda}")
torch.random.manual_seed(args.seed)
np.random.seed(args.seed)

# print(args.actlr, args.qlr, args.bufferSize, args.startLearning


class Trajectory:
    def __init__(self, observations, actions, rewards, dones, knownses):
        # print(observations[0])
        self.obs = np.stack(observations, axis=0)
        # print(self.obs.max(), self.obs.min()).
        self.actions = np.stack(actions, axis=0)
        self.dones = np.stack(dones, axis=0)
        self.knownses = np.stack(knownses, axis=0)
        self.rewards = np.stack(rewards, axis=0)
        # print("_________IN BUFFER________________", self.obs[0], self.actions[0], self.obs[1], self.rewards[0], sep="\n")
        if len(self.obs) != len(self.actions) + 1 != len(self.dones) != len(self.knownses) != len(self.rewards) + 1:
            print("ERROR IN BUFFER STORAGE")
        # print(self.dones.sum())
        # rewardSum = self.rewards.sum()
        # self.rewards = np.zeros_like(self.rewards)
        # self.rewards[-1] = rewardSum
        # print(self.rewards)
        # print(self.rewards)
        # print(self.obs.shape, self.actions.shape, self.rewards.shape)
        self.length = len(actions)

    def getElementWithContext(self, idx):
        # print(idx, self.length)
        if args.contextSAC:
            start = max(idx + 1 - args.context, 0)
        else:
            start = idx
        return {
            'obs': torch.tensor(self.obs[start:idx + 1]), 
            'actions': torch.tensor(self.actions[start:idx + 1]), 
            'nextobs': torch.tensor(self.obs[start+1:idx + 2]), 
            'dones': self.dones[idx + 1], 
            'rewards': self.rewards[idx]}
    
    def sampleWithContext(self, size):
        idxs = np.random.choice(self.length, size, replace = size>self.length)
        lens = np.where(idxs - args.context >=0, args.context, idxs + 1)
        # idxs = torch.tensor(idxs)

        # elements requiring context
        obs = torch.zeros([args.context, size, self.obs[0].shape[0]]).float()
        actions = torch.zeros([args.context, size, self.actions[0].shape[0]]).float()
        nextobs = torch.zeros([args.context, size, self.obs[0].shape[0]]).float()
        # efficiently fill tensors
        for x in range(1, args.context + 1):
            offset = args.context - x
            mask = np.where(idxs - offset >= 0, True, False)
            obs[x - 1, mask] += torch.tensor(self.obs[idxs[mask]]).float()
            actions[x - 1, mask] += torch.tensor(self.actions[idxs[mask]]).float()
            nextobs[x - 1, mask] += torch.tensor(self.obs[idxs[mask] + 1]).float()
        # elements not requiring context
        temp = obs == 0
        rewards = [np.mean(self.rewards)]
        dones = self.dones[idxs + 1]
        
        return {
            'obs': obs, 
            'actions': actions, 
            'nextobs': nextobs, 
            'dones': dones,
            'rewards': rewards,
            'lens': lens}
            # 'rewards': self.rewards}
    
    def retrieveStateFeatures(self, idx):
        mindx = max(idx - args.context, 0)
        obs = self.obs[mindx:idx + 1]
        acts = self.actions[mindx:idx + 1]
        labels = np.copy(self.obs[idx + 1]) - self.obs[idx]
        feats = np.concatenate([obs, acts], axis=1)
        knowns = self.knownses[idx + 1]
        
        return feats, labels, knowns
    
    def getElement(self, idx):
        # print(idx, self.length)
        return {
            'obs':self.obs[idx], 
            'actions': self.actions[idx], 
            'nextobs': self.obs[idx + 1], 
            'dones': self.dones[idx + 1], 
            'knowns': self.knownses[idx], 
            'nextknowns': self.knownses[idx + 1], 
            'rewards': self.rewards[idx]}
    
    def sample(self, size):
        idxs = np.random.choice(self.length, size, replace = size>self.length)
        # if size > self.length:
        #     mult = 1
        # else:
        #     mult = 1
        return {
            'obs': self.obs[idxs], 
            'actions': self.actions[idxs], 
            'nextobs': self.obs[idxs + 1], 
            'dones': self.dones[idxs + 1], 
            'knowns': self.knownses[idxs], 
            'nextknowns': self.knownses[idxs + 1], 
            'rewards': [np.mean(self.rewards)]}
            # 'rewards': self.rewards}



class Buffer:
    # keep track of max/min state component values observed
    maxStateVals = None
    minStateVals = None
    def __init__(self, numElements):
        self.n = numElements
        self.count = 0
        self.els = []
        self.splits = []

    def addElement(self, obs, act, rew, dones, knownses):
        tempObsMax = np.max(obs, axis=0)
        tempObsMin = np.min(obs, axis=0)
        if (self.count == 0):
            self.maxStateVals = tempObsMax
            self.minStateVals = tempObsMin
        else:
            self.maxStateVals = np.maximum(self.maxStateVals, tempObsMax)
            self.minStateVals = np.minimum(self.minStateVals, tempObsMin)
        e = Trajectory(obs, act, rew, dones, knownses)
        self.els.append(e)
        self.count += e.length

        if len(self.splits) > 0:
            self.splits.append(self.splits[-1] + len(act))
        else: 
            self.splits.append(len(act))
        if self.count > self.n:
            temp = self.els.pop(0)
            self.count -= temp.length
            self.splits = self.splits[1:]
            self.splits = list(map(lambda x: x - temp.length, self.splits))

    def sampleForSAC(self, size):
        idxs = np.random.choice(self.count, size, replace = size>self.count)
        idxs = sorted(idxs)
        i = 0
        toReturn = {}
        for idx in idxs:
            while (idx >= self.splits[i]):
                i += 1
            if i == 0:
                offset = 0
            else:
                offset = self.splits[i - 1]
            if (idx - offset) < 0:
                print("ERRORR!!!!!!! Buffer index offset wrong")
            temp = self.els[i].getElement(idx - offset)
            for key in temp.keys():
                if key in toReturn:
                    # print("HAS", i)
                    # print(toReturn[key].shape)
                    toReturn[key].append(temp[key])
                else:
                    # print("HASNOT", i)
                    toReturn[key] = [temp[key]]

        for key in ['obs', 'actions', 'nextobs']:
            toReturn[key] = torch.tensor(np.array(toReturn[key]))
        rrdfeats = torch.cat([toReturn['obs'], toReturn['actions'], toReturn['nextobs'] - toReturn['obs']], dim=-1).float()
        qfeats = torch.cat([toReturn['obs'], toReturn['actions']], dim=-1).float()
        actfeats = toReturn['obs'].float()
        # print("SAC:", rrdfeats.shape, qfeats.shape, actfeats.shape)
        return rrdfeats, qfeats, actfeats, torch.tensor(toReturn['dones']).int().unsqueeze(-1)
    
    def sampleForRRD(self, subLen, numSubs):
        idxs = np.random.choice(len(self.els), numSubs, replace = numSubs>len(self.els))
        toReturn = {}
        # print(idxs, "!!")
        for i in idxs:
            temp = self.els[i].sample(subLen)
            for key in temp.keys():
                if key in toReturn:
                    # print("HAS", i)
                    # print(toReturn[key].shape)
                    toReturn[key].append(temp[key])
                else:
                    # print("HASNOT", i)
                    toReturn[key] = [temp[key]]
                    # print(toReturn[key])
        for key in ['obs', 'actions', 'nextobs']:
            toReturn[key] = torch.tensor(np.array(toReturn[key]))
        feats = torch.cat([toReturn['obs'], toReturn['actions'], toReturn['nextobs'] - toReturn['obs']], dim=-1).float()
        rews = torch.tensor(np.array(toReturn['rewards'])).float()
        return feats, rews

    def sampleForOnlineDT(self, size):
        idxs = np.random.choice(self.count, size, replace = size>self.count)
        idxs = sorted(idxs)
        i = 0
        toReturn = {}
        for idx in idxs:
            while (idx >= self.splits[i]):
                i += 1
            if i == 0:
                offset = 0
            else:
                offset = self.splits[i - 1]
            if (idx - offset) < 0:
                print("ERRORR!!!!!!! Buffer index offset wrong")
            temp = self.els[i].getElementWithContext(idx - offset)
            for key in temp.keys():
                if key in toReturn:
                    # print("HAS", i)
                    # print(toReturn[key].shape)
                    toReturn[key].append(temp[key])
                else:
                    # print("HASNOT", i)
                    toReturn[key] = [temp[key]]
                    # print(toReturn[key])
        lens = [len(obs) for obs in toReturn['obs']]
        lens = torch.tensor(lens).int()
        # print(lens)
        for key in toReturn:
            if isinstance(toReturn[key][0], torch.Tensor):
                toReturn[key] = torch.nn.utils.rnn.pad_sequence(toReturn[key]).float()

        actions = toReturn['actions']
        obs = toReturn['obs']
        rewards = toReturn['rewards']
        # print("SAC:", rrdfeats.shape, qfeats.shape, actfeats.shape)
        return obs, actions, rewards

    def sampleForContextSAC(self, size):
        idxs = np.random.choice(self.count, size, replace = size>self.count)
        idxs = sorted(idxs)
        i = 0
        toReturn = {}
        for idx in idxs:
            while (idx >= self.splits[i]):
                i += 1
            if i == 0:
                offset = 0
            else:
                offset = self.splits[i - 1]
            if (idx - offset) < 0:
                print("ERRORR!!!!!!! Buffer index offset wrong")
            temp = self.els[i].getElementWithContext(idx - offset)
            for key in temp.keys():
                if key in toReturn:
                    # print("HAS", i)
                    # print(toReturn[key].shape)
                    toReturn[key].append(temp[key])
                else:
                    # print("HASNOT", i)
                    toReturn[key] = [temp[key]]
                    # print(toReturn[key])
        lens = [len(obs) for obs in toReturn['obs']]
        lens = torch.tensor(lens).int()
        # print(lens)
        for key in toReturn:
            if isinstance(toReturn[key][0], torch.Tensor):
                toReturn[key] = torch.nn.utils.rnn.pad_sequence(toReturn[key]).float()

        rrdfeats = torch.cat([toReturn['obs'], toReturn['actions'], toReturn['nextobs'] - toReturn['obs']], dim=-1).float()
        qfeats = torch.cat([toReturn['obs'], toReturn['actions']], dim=-1).float()
        actfeats = toReturn['obs']
        # print("SAC:", rrdfeats.shape, qfeats.shape, actfeats.shape)
        return rrdfeats, qfeats, actfeats, torch.tensor(toReturn['dones']).int().unsqueeze(-1), lens
    
    def sampleForContextRRD(self, subLen, numSubs):
        idxs = np.random.choice(len(self.els), numSubs, replace = numSubs>len(self.els))
        toReturn = {}
        # print(idxs, "!!")
        for i in idxs:
            temp = self.els[i].sampleWithContext(subLen)
            for key in temp.keys():
                if key in toReturn:
                    # print("HAS", i)
                    # print(toReturn[key].shape)
                    toReturn[key].append(temp[key])
                else:
                    # print("HASNOT", i)
                    toReturn[key] = [temp[key]]
                    # print(toReturn[key])
        for key in toReturn:
            if isinstance(toReturn[key][0], torch.Tensor):
                toReturn[key] = torch.stack(toReturn[key], dim=1)

        
        feats = torch.cat([toReturn['obs'], toReturn['actions'], toReturn['nextobs'] - toReturn['obs']], dim=-1).float()
        rews = torch.tensor(np.array(toReturn['rewards'])).float()
        lens = torch.tensor(np.array(toReturn['lens'])).int()
        # print("RRD:", feats.shape, rews.shape, lens.shape)
        # print ("FIRST RRD:", feats.shape)
        return feats, rews, lens 
    
    def sampleForStatePred(self, size):
        idxs = np.random.choice(self.count, size, replace = size > self.count)
        idxs = sorted(idxs)
        i = 0
        returnFeats, returnLabs, returnKnowns = [], [], []
        for idx in idxs:
            while (idx >= self.splits[i]):
                i += 1
            if i == 0:
                offset = 0
            else:
                offset = self.splits[i - 1]
            if (idx - offset) < 0:
                print("ERRORR!!!!!!! Buffer index offset wrong")
            tfeat, tlab, tknown = self.els[i].retrieveStateFeatures(idx - offset)
            returnFeats.append(torch.tensor(tfeat))
            returnLabs.append(tlab)
            returnKnowns.append(tknown)
        lengths = [len(feat) for feat in returnFeats]
        # shape is (sequence, batch, features)
        feats = torch.nn.utils.rnn.pad_sequence(returnFeats).float()
        # print("STATE:", feats.shape)
        return feats, np.stack(returnLabs, axis=0), np.stack(returnKnowns, axis=0), lengths
    
    def sampleForDTStatePred(self, size):
        idxs = np.random.choice(self.count, size, replace = size>self.count)
        idxs = sorted(idxs)
        i = 0
        returnObs, returnActs, returnLabs, returnKnowns = [], [], []
        for idx in idxs:
            while (idx >= self.splits[i]):
                i += 1
            if i == 0:
                offset = 0
            else:
                offset = self.splits[i - 1]
            if (idx - offset) < 0:
                print("ERRORR!!!!!!! Buffer index offset wrong")
            tobs, tact, tlab, tknown = self.els[i].retrieveStateDTFeatures(idx - offset)
            returnObs.append(torch.tensor(tobs))
            returnActs.append(torch.tensor(tact))
            returnLabs.append(tlab)
            returnKnowns.append(tknown)
        lengths = [len(feat) for feat in tobs]
        # shape is (sequence, batch, features)
        obs = torch.nn.utils.rnn.pad_sequence(returnObs).float()
        acts = torch.nn.utils.rnn.pad_sequence(returnActs).float()
        mask = torch.zeros_like(obs).float()
        for idx, l in enumerate(lengths):
            mask[:l, idx, :] = 1
        # print("STATE:", feats.shape)
        return obs, acts, np.stack(returnLabs, axis=0), np.stack(returnKnowns, axis=0), mask


def layer_init(layer, bias_const=0.1):
    nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class StateLSTMNetwork(nn.Module):
    def __init__(self, envs, hidden_size=args.hiddenSizeLSTM):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(envs.observation_space.shape[0] + envs.action_space.shape[0], self.hidden_size)
        self.attnlayer = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2)
        self.outlayer = nn.Linear(self.hidden_size, envs.observation_space.shape[0])
    
    def forward(self, x, mins=None, maxes=None):
        if isinstance(x, torch.Tensor) and x.dim() < 2:
            x.unsqueeze_(0)
        output, (H,C) = self.lstm(x)
        # only use last prediction to keep context length consistent
        if isinstance(output, torch.nn.utils.rnn.PackedSequence):
            output = torch.nn.utils.rnn.pad_packed_sequence(output)[0]
        output = torch.sigmoid(output)
        output = self.attnlayer(output, output, output)[0][-1]
        output = torch.relu(output)
        output = self.outlayer(output)
        if not mins is None:
            mins = torch.tensor(mins)
            maxes = torch.tensor(maxes)
            if args.cuda:
                mins = mins.cuda()
                maxes = maxes.cuda()
            output = torch.clamp(output, mins, maxes)
        return output
    
class StateNNNetwork(nn.Module):
    def __init__(self, envs, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.l1 = layer_init(nn.Linear(envs.observation_space.shape[0] + envs.action_space.shape[0], self.hidden_size))
        self.l2 = layer_init(nn.Linear(self.hidden_size, self.hidden_size))
        self.outlayer = layer_init(nn.Linear(self.hidden_size, envs.observation_space.shape[0]))
    
    def forward(self, x):
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x = torch.nn.utils.rnn.pad_packed_sequence(x)[0]
        if x.dim() == 3:
            x = x.reshape([x.shape[1], x.shape[0] * x.shape[2]])
        output = self.l1(x)
        output = F.relu(output)
        output = self.l2(output)
        output = F.relu(output)
        output = self.outlayer(output)
        return output


class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.input_shape = envs.observation_space.shape[0] + envs.action_space.shape[0]
        if (args.contextSAC):
            hiddenSize = args.hiddenSizeLSTM
            self.lstm = nn.LSTM(self.input_shape, hiddenSize)
        else:
            hiddenSize = 256
            self.fc1 = layer_init(nn.Linear(self.input_shape, hiddenSize))
            self.fc2 = layer_init(nn.Linear(hiddenSize, hiddenSize))
        self.fc_q = layer_init(nn.Linear(hiddenSize, 1))

    def forward(self, x):
        if (args.contextSAC):
            x, (H, C) = self.lstm(x)
        else:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x = torch.nn.utils.rnn.pad_packed_sequence(x)[0]
            x = x[-1, :, :]
        elif args.contextSAC:
            x = x[-1, :].unsqueeze(0)
        x = F.relu(x)
        q_vals = self.fc_q(x)
        return q_vals


class Actor(nn.Module):
    def __init__(self, envs: gym.Env):
        super().__init__()
        self.obs_shape = envs.observation_space.shape
        self.action_shape = envs.action_space.shape
        self.action_scale = torch.tensor((envs.action_space.high - envs.action_space.low) / 2.0).float()
        if args.cuda:
            self.action_scale = self.action_scale.cuda()
        if (args.contextSAC):
            hiddenSize = args.hiddenSizeLSTM
            self.lstm = nn.LSTM(self.obs_shape[0], hiddenSize)
        else:
            hiddenSize = 256
            self.fc1 = layer_init(nn.Linear(self.obs_shape[0], hiddenSize))
            self.fc2 = layer_init(nn.Linear(hiddenSize, hiddenSize))
        self.fc_mean_logdev = layer_init(nn.Linear(hiddenSize, 2*self.action_shape[0]))

    def forward(self, x):
        if (args.contextSAC):
            x, (H, C) = self.lstm(x)
        else:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x = torch.nn.utils.rnn.pad_packed_sequence(x)[0]
            x = x[-1, :, :]
        elif args.contextSAC:
            x = x[-1, :].unsqueeze(0)
        x = F.relu(x)
        temp = self.fc_mean_logdev(x)
        mean = temp[:, :self.action_shape[0]]
        logdev = temp[:, self.action_shape[0]:]
        logdev = torch.clamp(logdev, -20, 2)

        return mean, logdev
    
    def only_action(self, x, exploration=True):
        with torch.no_grad():
            mean, logdev = self(x)
            dev = torch.exp(logdev)
            if (exploration):
                policy_dist = Normal(loc=mean, scale=dev)
                samp = policy_dist.sample()
            else:
                samp = mean
            if (args.cuda):
                samp = samp.cuda()
            action = torch.tanh(samp) * self.action_scale
            return action

    def get_action(self, x, epsilon=1e-6, debug=False, exploration=False):
        mean, logdev = self(x)
        if (debug):
            print(logdev)
            print(x.shape, mean.shape, logdev.shape)
        dev = torch.max(logdev.exp(), .01*torch.ones_like(logdev))
        policy_dist = Normal(loc=mean, scale=dev)
        if (exploration):
            samp = policy_dist.rsample()
        else:
            samp = mean
        if (args.cuda):
            samp = samp.cuda()
        action = torch.tanh(samp) * self.action_scale
        logprob = policy_dist.log_prob(samp)
        # Enforcing Action Bound
        logprob -= torch.log(self.action_scale * (1 - action.pow(2)) + epsilon)
        logprob = logprob.sum(1, keepdim=True)
        # print(samp.shape, action.shape, logprob.shape)
        # exit()
        return action, logdev, logprob

class RRDModel(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.input_shape = 2 * envs.observation_space.shape[0] + envs.action_space.shape[0]
        if (args.contextSAC):
            contextSize = args.hiddenSizeLSTM
            self.lstm = nn.LSTM(self.input_shape, contextSize)
        else:
            contextSize = 256
            self.fc1 = layer_init(nn.Linear(self.input_shape, contextSize))
            self.fc2 = layer_init(nn.Linear(contextSize, contextSize))
        self.fc3 = layer_init(nn.Linear(contextSize, 1))

    def forward(self, x):
        if (args.contextSAC):
            out, (h, c) = self.lstm(x)
                # only use last prediction to keep context length consistent
        else:
            out = self.fc1(x)
            out = F.relu(out)
            out = self.fc2(out)
        if isinstance(out, torch.nn.utils.rnn.PackedSequence):
            out = torch.nn.utils.rnn.pad_packed_sequence(out)[0]
            out = out[-1, :, :]
        elif args.contextSAC:
            out = out[-1, :].unsqueeze(0)
        out = F.relu(out)
        out = self.fc3(out)
        return out
    

antParts = [[0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18],
    [5, 6, 19, 20],
    [7, 8, 21, 22],
    [9, 10, 23, 24],
    [11, 12, 25, 26]]

cheetahParts = [[0, 1, 8, 9, 10],
                [2, 11],
                [3, 12],
                [4, 13],
                [5, 14],
                [6, 15],
                [7, 16]]

humanParts = [[0, 1, 2, 3, 4, 22, 23, 24, 25, 26, 27],
              [5, 6, 7, 28, 29, 30],
              [8, 9, 10, 31, 32, 33],
              [11, 34],
              [12, 13, 14, 35, 36, 37],
              [15, 38],
              [16, 17, 39, 40],
              [18, 41],
              [19, 20, 42, 43],
              [21, 44]]

walkerParts = [[0, 1, 8, 9, 10],
               [2, 11],
               [3, 12],
               [4, 13],
               [5, 14],
               [6, 15],
               [7, 16]]

hopperParts = [[0, 1, 5, 6, 7],
               [2, 8],
               [3, 9],
               [4, 10]]

partsLookup = {"Ant-v2": antParts,
               "Ant-v3": antParts,
               "HalfCheetah-v2": cheetahParts,
               "Humanoid-v2": humanParts,
               "Hopper-v2": hopperParts,
               "Walker2d-v2": walkerParts}

def obsFilter(observation, numHidden, lastKnown, leftTilRandom):
    parts = partsLookup[args.env]
    if (numHidden > 0):
        if leftTilRandom <= 1:
            hiddenParts = np.random.choice(len(parts), numHidden, replace=False)
            known = np.ones_like(observation)
            leftReturn = args.consecHidden
            observation = np.copy(observation)
            for part in hiddenParts:
                observation[parts[part]] = 0
                known[parts[part]] = 0
        else:
            known = np.copy(lastKnown)
            observation = observation * known
            leftReturn = leftTilRandom - 1
    

        return observation, known, leftReturn
    # default
    return observation, np.ones_like(observation), leftTilRandom

def getStateBelief(observations, knowns, acts=None, statepred = None, mins=None, maxes=None):
    if (args.numHidden < 1) or (not args.statefill):
        return observations[-1]
    elif args.statepred:
        obsfeat = torch.tensor(np.concatenate([observations, acts], axis=-1)).float()
        obsfeat = obsfeat.unsqueeze_(1)
        if args.cuda:
            obsfeat = obsfeat.cuda()
        with torch.no_grad():
            toReturn = statepred(obsfeat, mins, maxes).squeeze()
        if args.cuda:
            toReturn = toReturn.cpu().detach().numpy()
        else:
            toReturn = toReturn.detach().numpy()
        return toReturn + observations[-1]
    observations = observations[-1]
    if len(observations.shape) > 2:
        print(observations.shape)
        print("OBS SHAPE INCORRECT!!!!")
    else:
        latest = observations[0]
        for x in range(1, len(observations)):
            # update latest knowledge
            latest = (knowns[x] * observations[x]) + ((1 - knowns[x]) * latest)
            # update observation with latest knowledge
            observations[x] = latest
    return observations


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env = gym.make(args.env)
env.seed(args.seed)
testenv = gym.make(args.env)
testenv.seed(args.seed)
actor = Actor(env)
qf1 = SoftQNetwork(env)
qf2 = SoftQNetwork(env)
qf1_target = SoftQNetwork(env)
qf2_target = SoftQNetwork(env)
rrder = RRDModel(env)
if (args.statepred):
    if args.statemodel == "nn":
        statepred = StateNNNetwork(env)
        statepred_target = StateNNNetwork(env)
    elif args.statemodel == "lstm":
        # exit()
        statepred = StateLSTMNetwork(env)
        statepred_target = StateLSTMNetwork(env)
    elif args.statemodel == "dt":
        statepred = DecisionTransformer(
            state_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            max_length=args.context,
            max_ep_len=1000,
            hidden_size=128,
            n_layer=3,
            n_head=1,
            n_inner=4*128,
            activation_function='relu',
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        statepred_target = DecisionTransformer(
            state_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            max_length=args.context,
            max_ep_len=1000,
            hidden_size=128,
            n_layer=3,
            n_head=1,
            n_inner=4*128,
            activation_function='relu',
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
else:
    statepred = None
    statepred_target = None
if args.cuda:
    actor = actor.cuda()
    qf1 = qf1.cuda()
    qf2 = qf2.cuda()
    qf1_target = qf1_target.cuda()
    qf2_target = qf2_target.cuda()
    rrder = rrder.cuda()
    if (args.statepred):
        statepred = statepred.cuda()
        statepred_target = statepred_target.cuda()
        statepred_target.load_state_dict(statepred.state_dict())
for p1, p2 in zip(qf1_target.parameters(), qf2_target.parameters()):
    p1.requires_grad = False
    p2.requires_grad = False
if (args.alpha_lr > 0):
    logalpha = torch.tensor(0).float()
    if (args.cuda):
        logalpha = logalpha.cuda()
    logalpha.requires_grad = True
    alpha_opt = optim.Adam([logalpha], lr=args.alpha_lr)
    alpha = torch.exp(logalpha)
else: 
    alpha = args.alpha


qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())

q_opt = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.qlr)
act_opt = optim.Adam(list(actor.parameters()), lr=args.actlr)
rrd_opt = optim.Adam(list(rrder.parameters()), lr=3e-4)
if args.statepred:
    stateopt = optim.Adam(list(statepred.parameters()), lr=args.statelr)

def evaluatePolicy(numRollouts=10):
    totalReward = 0
    totalLoss = 0
    for x in range(numRollouts):
        episodicReward = 0
        episodicLoss = 0
        done = False
        testobs = testenv.reset()
        hiddenLeft = args.consecHidden
        knowns = np.ones_like(testobs)
        obsbuff = [testobs]
        actbuff = []
        while not done:
            start = min(args.context, len(obsbuff))
            if args.contextSAC:
                obsfeat = torch.tensor(np.array(obsbuff[-start:])).float()
            else:
                obsfeat = torch.tensor(testobs).float().unsqueeze(0)
            if args.cuda:
                obsfeat = obsfeat.cuda()
            testaction = actor.only_action(obsfeat, exploration=False)
            if args.cuda:
                testaction = testaction.detach().cpu().numpy().squeeze()
            else:
                testaction = testaction.detach().numpy().squeeze()
            nextobsUn, testreward, done, _info = testenv.step(testaction)
            actbuff.append(testaction)
            nextObsPred = getStateBelief(obsbuff[-start:], knowns, actbuff[-start:], statepred, mins=buff.minStateVals, maxes=buff.maxStateVals)
            nextobs, knowns, hiddenLeft = obsFilter(nextobsUn, args.numHidden, knowns, hiddenLeft)
            testobs = (knowns * nextobs) + ((1 - knowns) * (nextObsPred))
            testObsIdxs = (1 - knowns) != 0
            # don't track timesteps where num hidden is 0 (first observation)
            if testObsIdxs.sum() > 0:
                # print(nextobsUn[testObsIdxs])
                testLoss = np.mean(np.square(nextObsPred[testObsIdxs] - nextobsUn[testObsIdxs]))
                episodicLoss += testLoss
            obsbuff.append(testobs)
            episodicReward += testreward
        totalLoss += episodicLoss / len(actbuff)
        totalReward += episodicReward
    return totalReward / numRollouts, totalLoss / numRollouts

if args.statepred:
    statemodel = args.statemodel
elif args.statefill:
    statemodel = "Fill"
else:
    statemodel = "NoFill"
if args.contextSAC:
    sacstr = f"{args.context}cSAC"
else:
    sacstr = "NoContext"
foldern = f"./saved_mujoco/{args.env}/{sacstr}/{statemodel}/"
if (args.logging and args.seed == 1):
    if not os.path.exists(f"{foldern}rewards"):
        os.makedirs(f"{foldern}rewards")
    if not os.path.exists(f"{foldern}actloss"):
        os.makedirs(f"{foldern}actloss")
    if not os.path.exists(f"{foldern}qfloss"):
        os.makedirs(f"{foldern}qfloss")
    if not os.path.exists(f"{foldern}rrdloss"):
        os.makedirs(f"{foldern}rrdloss")
    if not os.path.exists(f"{foldern}statepredloss"):
        os.makedirs(f"{foldern}statepredloss")
    if not os.path.exists(f"{foldern}statepredtestloss"):
        os.makedirs(f"{foldern}statepredtestloss")
    if not os.path.exists(f"{foldern}statepredunreducedloss"):
        os.makedirs(f"{foldern}statepredunreducedloss")
    if not os.path.exists(f"{foldern}staterange"):
        os.makedirs(f"{foldern}staterange")





rewardSet = []
rewardList = []
maxStateList = []
minStateList = []
testStateLossList = []
testStateLossUnreducedList = []
qflosslist = []
actlosslist = []
rrdlosslist = []
statelosslist = []
stateLossUnreducedList = []

buff = Buffer(args.bufferSize)

# init
observation = env.reset()
obs = [observation]
acts = []
dones = [0]
knowns = np.ones_like(observation)
knownses = [knowns]
rewards = []
numSteps = 0
hiddenLeft = 0

print(observation.shape)

actloss = None
qfloss = None

starttime = time.time()

for step in range(int(args.numSteps)):
    for envStep in range(args.envSteps):
        with torch.no_grad():
            start = min(args.context, len(obs))
            if args.contextSAC:
                obsfeat = torch.tensor(np.array(obs[-start:])).float()
            else:
                obsfeat = torch.tensor(obs[-1]).float().unsqueeze(0)
            if (envStep % 100) == 55 and step % 10 == 1:
                # print(obsfeat.max())
                None
            if args.cuda:
                obsfeat = obsfeat.cuda()
            action = actor.only_action(obsfeat)
            if args.cuda:
                action = action.detach().cpu().numpy().squeeze()
            else:
                action = action.detach().numpy().squeeze()
            nextObs, reward, done, info = env.step(action)
            acts.append(action)
            nextObsPred = getStateBelief(obs[-start:], knowns, acts[-start:], statepred_target, mins=buff.minStateVals, maxes=buff.maxStateVals)

            nextObs, knowns, hiddenLeft = obsFilter(nextObs, args.numHidden, knowns, hiddenLeft)
            # print(nextObsPred.shape, nextObs.shape)
            observation = (knowns * nextObs) + ((1 - knowns) * (nextObsPred))
            if (len(rewards) == 0):
                None
                # print("__________SAMPLED_________", observation, action, nextObs, reward, sep="\n")
            # observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
            knownses.append(knowns)
            rewards.append(reward)
            obs.append(observation)
            if done:
                if (info.get('TimeLimit.truncated', False)):
                    dones.append(0)
                else:
                    # print("Done!", len(acts))
                    dones.append(1)
                buff.addElement(obs, acts, rewards, dones, knownses)
                numSteps += len(acts)
                # init
                observation = env.reset()
                obs = [observation]
                acts = []
                dones = [0]
                knowns = np.ones_like(observation)
                knownses = [knowns]
                rewards = []
                hiddenLeft = 0
            else:
                dones.append(0)
    if (numSteps) >= args.startLearningState:
        # train state pred network
        if args.statepred:
            bsize = (args.train_batches * 256) // args.numBreaks
            for x in range(args.stateTrainMult*args.numBreaks):
                feats, labels, mask, lengths = buff.sampleForStatePred(bsize)
                feats = torch.nn.utils.rnn.pack_padded_sequence(feats, lengths, enforce_sorted=False)
                labels = torch.tensor(labels).float()
                mask = torch.tensor(mask).bool()
                if args.cuda:
                    feats = feats.cuda()
                    labels = labels.cuda()
                    mask = mask.cuda()
                preds = statepred(feats, buff.minStateVals, buff.maxStateVals)
                # preds = preds[mask]
                # labels = labels[mask]
                # print(preds.shape, labels.shape)
                stateLossUnreduced = (preds - labels)**2
                statePredLoss = (stateLossUnreduced * mask).mean()
                if args.cuda:
                    stateLossUnreduced = stateLossUnreduced.cpu()
                    mask = mask.cpu()
                stateLossUnreduced = stateLossUnreduced.detach().numpy()
                mask = mask.detach().numpy()
                stateLossUnreduced = np.average(stateLossUnreduced, axis=0, weights=mask)
                # print(statePredLoss.max().item())
                # statePredLoss = statePredLoss * mask
                stateopt.zero_grad()
                statePredLoss.backward()
                nn.utils.clip_grad_value_(statepred.parameters(), 1.0)
                stateopt.step()
    if (numSteps) >= args.startLearning:
        for trainstep in range(args.train_batches):
            # 1 update here for every environment step
            # train RRD network
            # 'obs':self.obs[idx], 
            # 'actions': self.actions[idx], 
            # 'nextobs': self.obs[idx + 1], 
            # 'dones': self.dones[idx + 1], 
            # 'knowns': self.knownses[idx], 
            # 'nextknowns': self.knownses[idx + 1], 
            # 'rewards': self.rewards[idx]
            if args.contextSAC:
                feats, labels, lens = buff.sampleForContextRRD(64, 4)
                # reshape to have only one batch dimension (necessary for LSTM)
                toReshape = [feats.shape[1], feats.shape[2]]
                lens = lens.reshape([feats.shape[1] * feats.shape[2]])
                feats = feats.reshape([feats.shape[0], feats.shape[1] * feats.shape[2], feats.shape[3]])
                # print(feats.shape, lens.shape)
                feats = torch.nn.utils.rnn.pack_padded_sequence(feats, lens, enforce_sorted=False)
            else:
                feats, labels = buff.sampleForRRD(64, 4)
            if args.cuda:
                feats = feats.cuda()
            
            rHat = rrder(feats).squeeze(-1)
            # reshape back to reconstruct sequences
            if len(rHat.shape) == 1:
                rHat = rHat.reshape(toReshape)
            # print(rHat.shape)
            episodicSums = torch.mean(rHat, dim=-1, keepdim=True)
            # print(episodicSums.shape, rewSamp.shape, feats.shape, rHat.shape)
            # print(rewSamp)
            # print(episodicSums.shape, rewSamp.shape)
            if args.cuda:
                labels = labels.cuda()
            rrdloss = F.mse_loss(episodicSums, labels)
            # print(rewSamp.shape, episodicSums.shape, rHat.shape)
            # if (step % 100) == 0:
            #     print(rrdloss.item())
            rrd_opt.zero_grad()
            rrdloss.backward()
            rrd_opt.step()

            # train Q-value networks
            if args.contextSAC:
                rdfeats, qfeats, actfeats, donesamp, lens = buff.sampleForContextSAC(256)
                actfeats = torch.nn.utils.rnn.pack_padded_sequence(actfeats, lens, enforce_sorted=False)
                rdfeats = torch.nn.utils.rnn.pack_padded_sequence(rdfeats, lens, enforce_sorted=False)
            else:
                rdfeats, qfeats, actfeats, donesamp = buff.sampleForSAC(256)

            if args.cuda:
                donesamp = donesamp.cuda()
                rdfeats = rdfeats.cuda()
                qfeats = qfeats.cuda()
                actfeats = actfeats.cuda()

            with torch.no_grad():
                rHat = rrder(rdfeats)
                # print(rdfeats.shape)
                nextActions, logdev, logprob = actor.get_action(actfeats, debug=False, exploration=True)
                nextFeats = qfeats.detach().clone()
                if args.contextSAC:
                    nextFeats[-1, :, -nextActions.shape[-1]:] -= nextFeats[-1, :, -nextActions.shape[-1]:]
                    nextFeats[-1, :, -nextActions.shape[-1]:] += nextActions
                    nextFeats = torch.nn.utils.rnn.pack_padded_sequence(nextFeats, lens, enforce_sorted=False)
                else:
                    # print(nextActions.shape, nextFeats.shape)
                    nextFeats[:, -nextActions.shape[-1]:] -= nextFeats[:, -nextActions.shape[-1]:]
                    nextFeats[:, -nextActions.shape[-1]:] += nextActions
                if args.cuda:
                    nextFeats = nextFeats.cuda()
                qtarg1 = qf1_target(nextFeats)
                qtarg2 = qf2_target(nextFeats)
                # print(qtarg1.shape)
                # print(nextFeats.shape, qtarg1.shape, logprob.shape)
                qtargmin = torch.min(qtarg1, qtarg2) - (alpha * logprob)
                # print(qtargmin.shape, logprob.shape, donesSamp.shape)
                qtarget = rHat + (args.gamma * (1 - donesamp) * qtargmin)
            
            if args.contextSAC:
                qfeatnow = torch.nn.utils.rnn.pack_padded_sequence(qfeats, lens, enforce_sorted=False)
            else:
                qfeatnow = qfeats.detach().clone()
            qf1vals = qf1(qfeatnow)
            # print(qfeatnow.shape, qf1vals.shape, qtarget.shape)
            # print(qf1vals.shape)
            qf2vals = qf2(qfeatnow)
            # if (trainstep % 99 == 0):
            #     print(qtarget.shape(), qf1vals.sum())
            # print(qf1vals.shape, qtarget.shape)
            # print(torch.mean(nextActions), np.mean(actSamp))
            qf1loss = F.mse_loss(qf1vals, qtarget)
            qf2loss = F.mse_loss(qf2vals, qtarget)
            qfloss = qf1loss + qf2loss
            # print(qfloss.item())
            q_opt.zero_grad()
            qfloss.backward()
            # nn.utils.clip_grad_value_(qf1.parameters(), 1)
            # nn.utils.clip_grad_value_(qf2.parameters(), 1)
            q_opt.step()

            # with torch.no_grad():
            #     qf1vals = qf1(feats)
            #     qf2vals = qf2(feats)
            #     qf1loss = F.mse_loss(qf1vals, qtarget)
            #     qf2loss = F.mse_loss(qf2vals, qtarget)
            #     qfloss = qf1loss + qf2loss
            #     print(qfloss.item())

            # train actor network

            actSamp, logdev, logprob = actor.get_action(actfeats, exploration=True)
            nextFeats = qfeats.detach().clone()
            # print(actSamp.shape)
            if args.contextSAC:
                with torch.no_grad():
                    nextFeats[-1, :, -actSamp.shape[-1]:] -= nextFeats[-1, :, -actSamp.shape[-1]:]
                nextFeats[-1, :, -actSamp.shape[-1]:] += actSamp
                nextFeats = torch.nn.utils.rnn.pack_padded_sequence(nextFeats, lens, enforce_sorted=False)
            else:
                with torch.no_grad():
                    nextFeats[:, -actSamp.shape[-1]:] -= nextFeats[:, -actSamp.shape[-1]:]
                nextFeats[:, -actSamp.shape[-1]:] += actSamp
            if args.cuda:
                nextFeats = nextFeats.cuda()
            for p1, p2 in zip(qf1.parameters(), qf2.parameters()):
                p1.requires_grad = False
                p2.requires_grad = False
            qf1vals = qf1(nextFeats)
            qf2vals = qf2(nextFeats)
            actloss = torch.mean((alpha * logprob) - torch.min(qf1vals, qf2vals))
            # print((alpha * logprob).mean(), torch.min(qf1vals, qf2vals).mean())
            # print(actloss.item())
            act_opt.zero_grad()
            actloss.backward()
            # print(actloss.item(), qfloss.item())
            # nn.utils.clip_grad_value_(actor.parameters(), 1)
            act_opt.step()
            for p1, p2 in zip(qf1.parameters(), qf2.parameters()):
                p1.requires_grad = True
                p2.requires_grad = True
            if (args.alpha_lr > 0):
                alpha_opt.zero_grad()
                with torch.no_grad():
                    multiplier = logprob - np.prod(env.action_space.shape)
                aloss = -1.0*(torch.exp(logalpha) * multiplier).mean()
                aloss.backward()
                alpha_opt.step()
                alpha = torch.exp(logalpha)

            # update target networks
            with torch.no_grad():
                for qt1w, qf1w in zip(qf1_target.parameters(), qf1.parameters()):
                    # print(qt1w.data.max(), qt1w.data.min())
                    qt1w.data.copy_(0.995 * qt1w.data + (.005 * qf1w.data))
                    # print(qt1w.data.max(), qt1w.data.min())
                    # print(qt1w.data)
                    # qt1w.data = qf1w.data
                for qt2w, qf2w in zip(qf2_target.parameters(), qf2.parameters()):
                    qt2w.data.copy_(0.995 * qt2w.data + (.005 * qf2w.data))
                    # qt2w.data = qf2w.data

                if args.statepred:
                    for spt, sp in zip(statepred_target.parameters(), statepred.parameters()):
                        spt.data.copy_(0.995 * spt.data + (.005 * sp.data))
            # if (step % 100) == 0:
            #     print(actloss.item())
    if numSteps >= args.startLearningState:
        if (step % 50) == 0 or (step == args.numSteps - 1):
            if args.statepred:
                statemodel = args.statemodel
            elif args.statefill:
                statemodel = "Fill"
            else:
                statemodel = "NoFill"
            if args.contextSAC:
                sacstr = f"{args.context}cSAC"
            foldern = f"./saved_mujoco/{args.env}/{sacstr}/{statemodel}/"
            fname = f"{args.numHidden}Hidden{args.seed}QLR{args.qlr}ALR{args.actlr}SLR{args.statelr}Start{args.startLearningState},{args.startLearning}HS{args.hiddenSizeLSTM}C{args.context}.csv"
            with torch.no_grad():
                testrew, testloss = evaluatePolicy()
                rewardList.append(testrew)
            if (args.statepred and args.logging):
                
                maxStateList.append(buff.maxStateVals)
                minStateList.append(buff.minStateVals)
                testStateLossList.append(testloss)
                statelosslist.append(statePredLoss.item())
                stateLossUnreducedList.append(stateLossUnreduced)

                np.savetxt(f"{foldern}/statepredunreducedloss/{fname}", stateLossUnreducedList, newline="\n", delimiter=",")
                np.savetxt(f"{foldern}/statepredtestloss/{fname}", testStateLossList, delimiter="\n")
                np.savetxt(f"{foldern}/statepredloss/{fname}", statelosslist, delimiter="\n")
                np.savetxt(f"{foldern}/staterange/MAX{fname}", maxStateList, newline="\n", delimiter=",")
                np.savetxt(f"{foldern}/staterange/MIN{fname}", minStateList, newline="\n", delimiter=",")
            if not actloss is None:
                if args.statepred:
                    print(f"Steps: {step * args.envSteps}, Time: {time.time() - starttime:.3f}s, Test rewards: {rewardList[-1]:.3f}, Actor loss: {actloss.item():.3e}, Q loss: {qfloss.item():.3e}, RRD loss: {rrdloss.item():.3e}, Alpha: {alpha:.3e}, StateLoss: {statePredLoss.item():.3e}, Test StateLoss: {testloss:.3e}")
                    # if (args.cuda):
                    #     print(f"GPU: {torch.cuda.max_memory_allocated(device=None)}")
                else:
                    print(f"Steps: {step * args.envSteps}, Time: {time.time() - starttime:.3f}s, Test rewards: {rewardList[-1]:.3f}, Actor loss: {actloss.item():.3e}, Q loss: {qfloss.item():.3e}, RRD loss: {rrdloss.item():.3e}, Alpha: {alpha:.3e}")    
            else:
                print(rewardList[-1])
            if (args.logging):
                
                np.savetxt(f"{foldern}/rewards/{fname}", rewardList, delimiter="\n")
                if not actloss is None:
                    actlosslist.append(actloss.item())
                    qflosslist.append(qfloss.item())
                    rrdlosslist.append(rrdloss.item())
                    np.savetxt(f"{foldern}/actloss/{fname}", actlosslist, delimiter="\n")
                    np.savetxt(f"{foldern}/qfloss/{fname}", qflosslist, delimiter="\n")
                    np.savetxt(f"{foldern}/rrdloss/{fname}", rrdlosslist, delimiter="\n")
