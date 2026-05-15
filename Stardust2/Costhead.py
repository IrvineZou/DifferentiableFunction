import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch_geometric.nn import MLP
import numpy as np

class FixedDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x

        if self.mask is None or self.mask.shape != x.shape:
            self.mask = (torch.rand_like(x) > self.p).float() / (1.0 - self.p)

        return x * self.mask



class optimizationNetwork(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(optimizationNetwork, self).__init__()
        self.input_dim = input_dim

        # Define the common network
        self.common_net = nn.Linear(input_dim,output_dim)
        self.norm1=nn.BatchNorm1d(output_dim)
        self.drop=FixedDropout(0)
        #self.actor_position = nn.Linear(input_dim, output_dim)  # logits for position selection
        #self.drop = FixedDropout(0.0001)
        self.reset()

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.float()
        output = self.common_net(x)
        #output = self.norm1(output)
        output = self.drop(output)
        #x=torch.relu(x)
        #x=self.drop(x)
        #logits_position = self.actor_position(x)
        #logits_position = self.drop(logits_position)
        return output#logits_position


    
class DifferentiableNetwork(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(DifferentiableNetwork, self).__init__()
        self.optimizednet=optimizationNetwork(input_dim,output_dim)
        #self.capsnetwork=CapsNetwork(input_dim,output_dim)
        self.dual=nn.Parameter(torch.ones(output_dim))

    def setEnv(self,env):
        self.env=env
 
    def computeDelay(self,actions):
        totals=0
        self.env.reset(True)
        #pdb.set_trace()
        for i in range(actions.size(0)):
            t=actions.cpu().numpy()
            tu=(i,t[i])
            pdb.set_trace()
            next_observation,reward,done,_,_=self.env.steptime(tu,False)
            totals+=reward
        #print("total throughput: ",totals)
        return totals

    def forward(self,x,w,caps,delay):
        #pdb.set_trace()
        P = self.optimizednet(x)                 # [N, D]
        P = torch.softmax(P, dim=-1)             # [N, D] assignment probs
        P_min = torch.amin(P, dim=1, keepdim=True)
        P_max = torch.amax(P, dim=1, keepdim=True)
        P_norm = (P - P_min) / (P_max - P_min)
        #the computation graph has been cut
        delays=torch.ones(P.size())*delay.unsqueeze(1)
        #pdb.set_trace()
        chip_load = torch.sum(delay.unsqueeze(1) * P_norm, dim=0)   # [D]
        f = torch.max(chip_load)
        #f=torch.logsumexp(delays*P,1)
        #f=torch.logsumexp(f,0)
        usage = torch.sum(P_norm * w.unsqueeze(1),dim=0)
        caps = caps.to(torch.float32)

        g = torch.relu((usage - caps)) / (caps.mean().clamp_min(1e-6))

        dual = torch.nn.functional.softplus(self.dual)
        lagrangian = f + 0.1*(dual * g).mean()

        loss_primal = lagrangian
        return (delays*P_norm),loss_primal,g
    




    def updatedual(self,g):
        with torch.no_grad():
            self.dual[:] = torch.clamp(self.dual + 1e-4 * g, min=0.0)
