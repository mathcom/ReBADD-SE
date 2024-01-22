import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import selfies as sf
from selfies.exceptions import SMILESParserError
from rdkit import Chem


def relu(x):
    return np.maximum(0, x)

    
class REINFORCE(object):
    def __init__(self, data, generator, reward_ft, tau_init=0.3, zeta=0.99):
        super(REINFORCE, self).__init__()
        self.data      = data
        self.generator = generator
        self.reward_ft = reward_ft
        
        ## constants
        self.psi = np.log(self.generator.output_size) ## for entropy reguarlization
        self.tau = tau_init ## reward requirement for trainable episodes
        self.xi = tau_init
        self.zeta = zeta


    def get_reward(self, sel):
        rwd = 0.
        try:
            smi = sf.decoder(sel)
            rwd = self.reward_ft(smi)
        except SMILESParserError:
            pass
        except sf.DecoderError:
            pass
        except sf.EncoderError:
            pass
        return rwd

    
    def evaluate(self, *args, **kwargs):
        return self.generator.evaluate(*args, **kwargs)
        
    
    def policy_gradient(self, n_batch=16, n_sample=16,
                        alpha=1., grad_clipping=None, verbose=0):
        ################################################
        ## Episode generation
        ################################################
        latentvectors, trajectories, rewards, rewards_norm = self.episode_sampling(n_batch, n_sample)
        n_trajectories = len(trajectories)
        
        ################################################
        ## Reinforcement Learning
        ################################################
        rl_loss = 0.
        reconstruction_loss = 0.
        entropy_loss = 0.
        total_reward = 0.
        
        if len(trajectories) == 0:
            print('[WARNING] No trajectory!!!')
        else:
            ## Set to train mode
            self.generator.train()
            
            ## Initialize gradients
            self.generator.optimizer.zero_grad()

            ## For each episode
            for z, trajectory, rwd, r in zip(latentvectors, trajectories, rewards, rewards_norm):
                
                ## Convert the char sequence into tensor
                inp = self.data.char_tensor(trajectory)

                ## Get initial hidden states
                hidden_0 = self.generator.init_hidden(z) # hidden_0.shape = (nlayer, 1, hidden)
                hidden = torch.tanh(hidden_0) # hidden.shape = (nlayer, 1, hidden)
                hidden_control = torch.sigmoid(hidden_0) # hidden_control.shape = (nlayer, 1, hidden)
                
                ## Initializing stack_memory
                stack = self.generator.init_stack() if self.generator.has_stack else None # stack.shape = (1, depth, width)

                ## Initialize loss values
                reconstruction_loss_ = 0.
                entropy_loss_ = 0.
                
                for c in range(len(inp)-1):
                    ## generator
                    output, hidden, stack = self.generator(inp[c], hidden, stack)
                    '''
                    output.shape = (1, vocab)
                    hidden.shape = (nlayer, 1, hiddensize)
                    stack.shape = (1, depth, width)
                    '''
                    log_probs = F.log_softmax(output, dim=1)
                    probs = F.softmax(output, dim=1)

                    ## policy entropy reguarlization
                    entropy_reg = self.psi + (probs[0,:] * log_probs[0,:]).sum()

                    ## loss
                    top_i = inp[c+1]
                    reconstruction_loss_ = reconstruction_loss_ - (log_probs[0, top_i] * r)
                    entropy_loss_ = entropy_loss_ + alpha * entropy_reg
                    
                    ## Joint for dependency on the latent vector
                    hidden = hidden * hidden_control

                ## average
                reconstruction_loss_ = reconstruction_loss_ / (len(inp) - 1)
                entropy_loss_ = entropy_loss_ / (len(inp) - 1)
                
                ## accumulate
                rl_loss = rl_loss + (reconstruction_loss_ + entropy_loss_)
                reconstruction_loss = reconstruction_loss + reconstruction_loss_
                entropy_loss = entropy_loss + entropy_loss_
                total_reward = total_reward + rwd

            ## Average over batches
            total_reward = total_reward / n_trajectories
            rl_loss = rl_loss / n_trajectories
            reconstruction_loss = reconstruction_loss / n_trajectories
            entropy_loss = entropy_loss / n_trajectories

            ## Doing backward pass and parameters update
            rl_loss.backward()
            if grad_clipping is not None:
                for param in self.generator.params_optimizer:
                    torch.nn.utils.clip_grad_norm_(param.get('params'), grad_clipping)
            self.generator.optimizer.step()

            ## Tensor to Constant 
            rl_loss = rl_loss.item()
            reconstruction_loss = reconstruction_loss.item()
            entropy_loss = entropy_loss.item()
        
        ## Increase tau
        self._update_tau(total_reward)
        
        return total_reward, rl_loss, reconstruction_loss, entropy_loss


    def sample_latent_vectors(self, batch_size=1):
        return self.generator.sample_latent_vectors()


    def episode_sampling(self, n_batch, n_sample):
        
        latentvectors = []
        trajectories = []
        rewards = []
        rewards_norm = []
    
        ## Set evaluation mode
        self.generator.eval()
        
        while len(trajectories) == 0:
            for _ in range(n_batch):
                ## Sample a latent vector
                z = self.sample_latent_vectors()
                
                ## For each latent vector
                for _ in range(n_sample):

                    ## Generate a SELFIES string
                    sel, _, trajectory = self.generator.evaluate(self.data, z=z, return_z=True, greedy=False)
                    
                    ## Trim start_token & end_token
                    sel = sel.replace(self.data.start_token, '').replace(self.data.end_token, '')
                    
                    ## Reward calculateion
                    rwd = self.get_reward(sel)
                    r = relu(rwd - self.tau)
                    if r > 0 and trajectory not in trajectories:
                        latentvectors.append(z)
                        trajectories.append(trajectory)
                        rewards.append(rwd)
                        rewards_norm.append(r)
                        
        return latentvectors, trajectories, rewards, rewards_norm


    def _update_tau(self, rwd):
        if rwd > self.xi:
            self.xi = rwd
            self.tau = self.zeta * self.tau + (1. - self.zeta) * rwd


class REINFORCE_SCST(REINFORCE):
    def __init__(self, data, generator, reward_ft, tau_init=0.3, zeta=0.99):
        super(REINFORCE_SCST, self).__init__(data, generator, reward_ft)
        
        
    def get_z_baseline(self, verbose=0):
        '''
        Reference:
        Rennie, Steven J., et al. "Self-critical sequence training for image captioning."
        Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR). 2017.
        '''
        self.generator.eval()
        r = 0.
        
        with torch.no_grad():
        
            while r < 1e-5:
                z = self.sample_latent_vectors()
                sel = self.generator.evaluate(self.data, z=z, return_z=False, greedy=True)
                sel = sel.replace(self.data.start_token, '').replace(self.data.end_token, '')
                rwd = self.get_reward(sel)
                if rwd > 0:
                    r = rwd

        return z, r
        
        
    def episode_sampling(self, n_batch, n_sample):
        latentvectors = []
        trajectories = []
        rewards = []
        rewards_norm = []
    
        ## Set evaluation mode
        self.generator.eval()
        
        while len(trajectories) == 0:
            for _ in range(n_batch):
                ## Sample a latent vector
                z, b = self.get_z_baseline() # b: baseline
                
                ## For each latent vector
                for _ in range(n_sample):
                    
                    ## Generate a SELFIES string
                    sel, _, trajectory = self.generator.evaluate(self.data, z=z, return_z=True, greedy=False)
                    
                    ## Trim start_token & end_token
                    sel = sel.replace(self.data.start_token, '').replace(self.data.end_token, '')
                    
                    ## Reward calculateion
                    rwd = self.get_reward(sel)
                    r = relu(rwd - b)
                    if rwd > self.tau and r > 0 and trajectory not in trajectories:
                        latentvectors.append(z)
                        trajectories.append(trajectory)
                        rewards.append(rwd)
                        rewards_norm.append(r)
                        
        return latentvectors, trajectories, rewards, rewards_norm
        

class REINFORCE_SCST_OFFPOLICY(REINFORCE_SCST):
    def __init__(self, data, generator, reward_ft, tau_init=0.3, zeta=0.99):
        super(REINFORCE_SCST_OFFPOLICY, self).__init__(data, generator, reward_ft)
        self.behavior = self.generator.copy()
    
    
    def update_behavior(self):
        self.behavior.load_state_dict(self.generator.state_dict())
    
    
    def get_z_baseline(self, n_sample, verbose=0):

        self.generator.eval()
        n_sample = max(2, n_sample)
        r = 0.
        
        with torch.no_grad():
        
            while r < 1e-8:
                z = self.sample_latent_vectors()
                
                rewards = []
                
                for _ in range(n_sample):
                    sel = self.generator.evaluate(self.data, z=z, return_z=False, greedy=False)
                    sel = sel.replace(self.data.start_token, '').replace(self.data.end_token, '')
                    rwd = self.get_reward(sel)
                    if rwd > 0:
                        rewards.append(rwd)
                    
                if len(rewards) >= 2:
                    r = np.mean(rewards)

        return z, r
        
        
    def episode_sampling(self, n_batch, n_sample):
        latentvectors = []
        trajectories = []
        rewards = []
        rewards_norm = []
    
        ## Set evaluation mode
        self.generator.eval()
        
        while len(trajectories) == 0:
            for _ in range(n_batch):
                ## Sample a latent vector
                z, b = self.get_z_baseline(n_sample) # b: baseline
                
                ##################################
                ## Behavior policy
                ##################################
                for _ in range(n_sample):
                    
                    ## Generate a SELFIES string
                    sel, _, trajectory = self.behavior.evaluate(self.data, z=z, return_z=True, greedy=False)
                    
                    ## Trim start_token & end_token
                    sel = sel.replace(self.data.start_token, '').replace(self.data.end_token, '')
                    
                    ## Reward calculateion
                    rwd = self.get_reward(sel)
                    r = relu(rwd - b)
                    if rwd > self.tau and r > 0 and trajectory not in trajectories:
                        latentvectors.append(z)
                        trajectories.append(trajectory)
                        rewards.append(rwd)
                        rewards_norm.append(r)
                        
                ##################################
                ## Target policy
                ##################################
                for _ in range(n_sample):
                    
                    ## Generate a SELFIES string
                    sel, _, trajectory = self.generator.evaluate(self.data, z=z, return_z=True, greedy=False)
                    
                    ## Trim start_token & end_token
                    sel = sel.replace(self.data.start_token, '').replace(self.data.end_token, '')
                    
                    ## Reward calculateion
                    rwd = self.get_reward(sel)
                    r = relu(rwd - b)
                    if rwd > self.tau and r > 0 and trajectory not in trajectories:
                        latentvectors.append(z)
                        trajectories.append(trajectory)
                        rewards.append(rwd)
                        rewards_norm.append(r)
                        
        return latentvectors, trajectories, rewards, rewards_norm