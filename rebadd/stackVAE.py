"""
This class implements generative recurrent neural network with augmented memory
stack as proposed in https://arxiv.org/abs/1503.01007
There are options of using LSTM or GRU, as well as using the generator without
memory stack.
"""
import time
import numpy as np
import selfies as sf
from tqdm import trange
from random import *
from rdkit.Chem import MolFromSmiles

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class StackAugmentedVAE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, latent_size=None,
                 n_layers=1, has_stack=False, max_seqlen=150,
                 stack_width=None, stack_depth=None, use_cuda=None, device=None,
                 optimizer_instance=torch.optim.Adadelta, lr=1e-4):

        super(StackAugmentedVAE, self).__init__()
        
        ## Model parameters
        self.has_stack = has_stack
        self.stack_width = stack_width if self.has_stack else 0
        self.stack_depth = stack_depth if self.has_stack else 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.latent_size = hidden_size if latent_size is None else latent_size
        self.n_layers = n_layers
        self.rnn_input_size = hidden_size + stack_width if self.has_stack else hidden_size
        self.max_seqlen = max_seqlen
        
        ## Model build - stack layers
        if self.has_stack:
            self.stack_controls_layer = nn.Linear(in_features=self.hidden_size * self.n_layers, out_features=3)
            self.stack_input_layer = nn.Linear(in_features=self.hidden_size * self.n_layers, out_features=self.stack_width)

        ## Model build - RNN layer
        self.enc = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, bidirectional=True)
        self.rnn = nn.GRU(self.rnn_input_size, self.hidden_size, self.n_layers, bidirectional=False)
        
        ## Model build - VAE layer
        self.hidden2mu = nn.Linear(2 * self.n_layers * self.hidden_size, self.latent_size)
        self.hidden2logvar = nn.Linear(2 * self.n_layers * self.hidden_size, self.latent_size)
        self.z2hidden = nn.Linear(self.latent_size, self.n_layers * self.hidden_size)
        
        ## Model build - Embedding layer
        self.encoder = nn.Embedding(self.input_size, self.hidden_size)
        
        ## Model build - Output layer
        self.decoder = nn.Linear(self.hidden_size, self.output_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        
        ## GPU configuration
        self.use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
        self.device = self.set_device() if device is None else device
        self.to(device)

        ## Loss setting
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        
        ## Optimizer setting
        self.lr = lr
        self.optimizer_instance = optimizer_instance
        
        self.params_optimizer_enc = [{'params':self.enc.parameters()},
                                     {'params':self.hidden2mu.parameters(), 'lr':lr*0.1},
                                     {'params':self.hidden2logvar.parameters(), 'lr':lr*0.1}]
        self.params_optimizer = [{'params':self.rnn.parameters()},
                                 {'params':self.encoder.parameters()},
                                 {'params':self.decoder.parameters()},
                                 {'params':self.z2hidden.parameters()}]
        if self.has_stack:
            self.params_optimizer += [{'params':self.stack_controls_layer.parameters()},
                                      {'params':self.stack_input_layer.parameters()}]
        
        self.optimizer = self.optimizer_instance(self.params_optimizer, lr=lr)
        self.optimizer_enc = self.optimizer_instance(self.params_optimizer_enc, lr=lr)


    def set_device(self):
        return torch.device('cuda') if self.use_cuda else torch.device('cpu')


    def load_model(self, path):
        weights = torch.load(path)
        self.load_state_dict(weights)


    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def change_lr(self, new_lr):
        self.optimizer = self.optimizer_instance(self.params_optimizer, lr=new_lr)
        self.optimizer_enc = self.optimizer_instance(self.params_optimizer_enc, lr=new_lr)
        self.lr = new_lr


    def forward(self, inp, hidden, stack):
        '''
        inp.shape = (1,)
        hidden.shape = (nlayer, 1, hiddensize)
        stack.shape = (1, depth, width)
        '''
        inp = self.encoder(inp.view(1, -1)) # inp.shape = (1, 1, hidden)
        if self.has_stack:
            ## make an input for stack
            hidden_ = hidden # hidden_.shape = (nlayer, 1, hiddensize)
            hidden_2_stack = hidden_.permute(1, 0, 2).view(1, -1) # hidden_2_stack.shape = (1, nlayer * hiddensize)
            
            ## compute stack controls
            stack_controls = self.stack_controls_layer(hidden_2_stack) # stack_controls.shape = (1, 3)
            stack_controls = F.softmax(stack_controls, dim=1) # stack_controls.shape = (1, 3)
            
            ## compute stackable information
            stack_input = self.stack_input_layer(hidden_2_stack.unsqueeze(0)) # stack_input.shape = (1, 1, width)
            stack_input = torch.tanh(stack_input) # stack_input.shape = (1, 1, width)
            
            ## put the stackable information
            stack = self.stack_augmentation(stack_input, stack, stack_controls) # stack.shape = (1, depth, width)
            stack_top = stack[:, :1, :]  # stack.shape = (1, 1, width)
            
            ## make an input for RNN by incorporating stack information
            inp = torch.cat((inp, stack_top), dim=2) # inp.shape = (1, 1, hidden + width)
        
        ## compute RNN
        output, next_hidden = self.rnn(inp, hidden) # output.shape = (1, 1, hidden), next_hidden.shape = (nlayer, 1, hiddensize)
        output = self.decoder(output.view(1, -1)) # output.shape = (1, vocab)
        return output, next_hidden, stack


    def stack_augmentation(self, input_val, prev_stack, controls):
        '''
        input_val.shape = (1, 1, width)
        prev_stack.shape = (1, depth, width)
        controls.shape = (1, 3)
        '''
        controls = controls.unsqueeze(-1).unsqueeze(-1) # controls.shape = (1, 3, 1, 1)
        ## stack zero padding
        zeros_at_the_bottom = torch.zeros((1, 1, self.stack_width), device=self.device) # zeros_at_the_bottom.shape = (1, 1, width)
        
        ## stack controls
        a_push = controls[:, 0] # a_push.shape = (1, 1, 1)
        a_pop  = controls[:, 1] # a_pop.shape  = (1, 1, 1)
        a_noop = controls[:, 2] # a_noop.shape = (1, 1, 1)
        
        ## stack_up : new stack (this will be pushed)
        stack_up   = torch.cat((input_val, prev_stack[:, :-1]), dim=1) # stack_up.shape = (1, depth, width)
        
        ## stack_down : old stack (this will be poped)
        stack_down = torch.cat((prev_stack[:, 1:], zeros_at_the_bottom), dim=1) # stack_down.shape = (1, depth, width)
        
        ## stack update (do push & pop)
        new_stack  = (a_noop * prev_stack) + (a_push * stack_up) + (a_pop * stack_down) # new_stack.shape = (1, depth, width)
        return new_stack 


    def init_hidden(self, z): # z.shape = (1, latent_size)
        h_0 = self.z2hidden(z) # h_0.shape = (1, n_layers * hidden_size)
        h_0 = h_0.view(1, self.n_layers, self.hidden_size) # h_0.shape = (1, n_layers, hidden_size)
        h_0 = h_0.permute(1, 0, 2) # h_0.shape = (n_layers, 1, hidden_size)
        return h_0


    def init_stack(self):
        return torch.zeros((1, self.stack_depth, self.stack_width), device=self.device) # (1, depth, width)


    def calc_latent(self, inp): # inp.shape = (seqlen, )
        inp = inp.view(1, -1) # inp.shape = (1, seqlen)
        emb = self.encoder(inp) # emb.shape = (1, seqlen, hidden)
        emb = emb.permute(1, 0, 2) # emb.shape = (seqlen, 1, hidden)
        _, h_n = self.enc(emb) # h_n.shape = (2 * n_layer, 1, hidden)
        h_n = h_n.permute(1, 0, 2) # h_n.shape = (1, 2 * n_layer, hidden)
        h_n = h_n.view(1, -1) # h_n.shape = (1, 2 * n_layer * hidden)
        mu = self.hidden2mu(h_n) # mu.shape = (1, latent)
        logvar = self.hidden2logvar(h_n) # logvar.shape = (1, latent)
        return mu, logvar 
        
        
    def reparameterization(self, mu, logvar): # mu.shape = (1, latent), logvar.shape = (1, latent)
        eps = torch.randn_like(mu, device=self.device) # eps.shape = (1, latent)
        z = eps * torch.exp(0.5 * logvar) + mu # z.shape = (1, latent)
        return z
        
        
    def sample_latent_vectors(self, batch_size=1):
        return torch.randn((batch_size, self.latent_size), device=self.device)
        
        
    def train_step(self, data, batch_size, beta=1.):
        self.train()
        
        ## initialize an optimizer
        self.optimizer.zero_grad()
        self.optimizer_enc.zero_grad()
        loss = 0.
        loss_rec = 0.
        loss_kld = 0.
        
        for _ in range(batch_size):
            inp, target = data.random_training_set() # inp.shape = (seqlen, ), target.shape = (seqlen, )
            
            ## encoder
            mu, logvar = self.calc_latent(inp) # mu.shape = (1, latent), logvar.shape = (1, latent)
            z = self.reparameterization(mu, logvar) # z.shape = (1, latent)
            
            ## initialize rnn_cell
            hidden_0 = self.init_hidden(z) # hidden_0.shape = (nlayer, 1, hidden)
            hidden = torch.tanh(hidden_0) # hidden.shape = (nlayer, 1, hidden)
            hidden_control = torch.sigmoid(hidden_0) # hidden_control.shape = (nlayer, 1, hidden)
            
            ## initialize stack_memory
            stack = self.init_stack() if self.has_stack else None # stack.shape = (1, depth, width)

            ## reconstruction loss
            loss_rec_ = 0.
            for c in range(len(inp)):
                output, hidden, stack = self(inp[c], hidden, stack)
                '''
                output.shape = (1, vocab)
                hidden.shape = (nlayer, 1, hiddensize)
                stack.shape = (1, depth, width)
                '''
                loss_rec_ = loss_rec_ + self.criterion(output, target[c].unsqueeze(0))
                
                ## memory in hidden state is controlled by the latent vector
                hidden = hidden * hidden_control
            
            ## average
            loss_rec_ = loss_rec_ / len(inp)
            
            ## KL divergence loss
            loss_kld_ = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
            
            ## accumulate losses
            loss_rec = loss_rec + loss_rec_
            loss_kld = loss_kld + loss_kld_

        ## average over batch
        loss_rec = loss_rec / batch_size
        loss_kld = loss_kld / batch_size
        loss = loss_rec + loss_kld * beta
            
        ## backpropagation
        loss.backward()
        self.optimizer.step()
        self.optimizer_enc.step()

        self.eval()
        return loss.item(), loss_rec.item(), loss_kld.item()
    
    
    def evaluate(self, data, z=None, return_z=False, temperature=None, greedy=False):
        ## evaluation mode
        self.eval()
        
        ## init - string
        prime_str = data.start_token
        end_token = data.end_token
        
        ## init - tensor
        if z is None:
            z = self.sample_latent_vectors() # z.shape = (1, latent)
        hidden_0 = self.init_hidden(z) # hidden_0.shape = (nlayer, 1, hidden)
        hidden = torch.tanh(hidden_0) # hidden.shape = (nlayer, 1, hidden)
        hidden_control = torch.sigmoid(hidden_0) # hidden_control.shape = (nlayer, 1, hidden)

        ## initialize stack_memory
        stack = self.init_stack() if self.has_stack else None

        ## make a starting token
        char_list = [prime_str]
        prime_input = data.char_tensor(prime_str)
        new_sample = prime_str
        inp = prime_input[-1]

        ## Begin the string generation
        for p in range(self.max_seqlen):
            output, hidden, stack = self.forward(inp, hidden, stack)

            if greedy:
                _, top_i = torch.topk(output, 1, dim=-1) # top_i.shape = (1, 1, 1)
                top_i = top_i.item()
            else:
                ## Sample from the network as a multinomial distribution
                if temperature is None:
                    probs = torch.softmax(output, dim=1)
                else:
                    probs = torch.softmax(output / temperature, dim=1)

                ## insert error handler
                if np.isnan(probs.cpu().detach().numpy()).all():
                    top_i = randint(0, len(data.all_characters)-1) # randomly selected index
                    print("all the elements of probs are NaN, so randomly select an index: " + str(top_i))
                else:
                    top_i = torch.multinomial(probs.view(-1), 1)[0].item()

            ## Add predicted character to string
            predicted_char = data.vocabs[top_i]
            new_sample += predicted_char
            char_list.append(predicted_char)

            ## sentense generation is complete?
            if predicted_char == end_token:
                break
            else:
                ## prepare next inputs
                inp = data.char_tensor(predicted_char)
                ## memory in hidden state is controlled by the latent vector
                hidden = hidden * hidden_control

        ## Check end token
        if char_list[-1] != end_token:
            char_list.append(end_token)
            new_sample += end_token
        
        ## reset to training mode
        self.train()
        
        if return_z:
            return new_sample, z, char_list
        else:
            return new_sample


    def fit(self, data, n_iterations, batch_size=10, print_every=100, ckpt_every=10,
            model_path=None, losses_path=None):
        
        self.loss_vae_list = []
        self.loss_rec_list = []
        self.loss_kld_list = []
        self.beta_list = []
        
        start = time.time()
        for epoch in trange(1, n_iterations + 1, desc='Training in progress...'):
            beta = min(max(0., epoch / n_iterations), 1.)
            loss_vae, loss_rec, loss_kld = self.train_step(data, batch_size, beta=beta) 
            self.loss_vae_list.append(loss_vae)
            self.loss_rec_list.append(loss_rec)
            self.loss_kld_list.append(loss_kld)
            self.beta_list.append(beta)

            if epoch % print_every == 0:
                loss_vae_avg = np.mean(self.loss_vae_list[-print_every:])
                loss_rec_avg = np.mean(self.loss_rec_list[-print_every:])
                loss_kld_avg = np.mean(self.loss_kld_list[-print_every:])
                log = f'[{epoch:05d} ({epoch/n_iterations*100:.1f}%) {time_since(start)}]'
                log += f', Loss_vae:{loss_vae_avg:.3f}'
                log += f', Loss_rec:{loss_rec_avg:.3f}'
                log += f', Loss_kld:{loss_kld_avg:.3f}'
                log += f', Beta:{beta:.3f}'
                print(log)
                log_selfies = self.evaluate(data=data).replace(data.start_token, '').replace(data.end_token, '')
                print(f'selfies: {log_selfies}\nsmiles: {sf.decoder(log_selfies)}')

            if epoch % ckpt_every == 0:
                if model_path is not None:
                    self.save_model(model_path + f".{epoch:05d}")
                if losses_path is not None:
                    with open(losses_path, 'w') as fout:
                        losses_header = "\t".join(["LOSS_VAE",
                                                   "LOSS_RECONSTRUCTION",
                                                   "LOSS_KLDIVERGENCE",
                                                   "BETA"])
                        fout.write(f"{losses_header}\n")
                        for loss_vae, loss_rec, loss_kld in zip(self.loss_vae_list, self.loss_rec_list, self.loss_kld_list):
                            losses_line = "\t".join([f"{loss_vae:.6f}",
                                                     f"{loss_rec:.6f}",
                                                     f"{loss_kld:.6f}",
                                                     f"{beta:.6f}"])
                            fout.write(f"{losses_line}\n")
                
        return {'LOSS_VAE':self.loss_vae_list,
                'LOSS_RECONSTRUCTION':self.loss_rec_list,
                'LOSS_KLDIVERGENCE':self.loss_kld_list,
                'BETA':self.beta_list}

    def copy(self):
        kwargs_generator = {"input_size"         : self.input_size,
                            "output_size"        : self.output_size,
                            "max_seqlen"         : self.max_seqlen,
                            "hidden_size"        : self.hidden_size,
                            "latent_size"        : self.latent_size,
                            "n_layers"           : self.n_layers,
                            "has_stack"          : self.has_stack,
                            "stack_width"        : self.stack_width,
                            "stack_depth"        : self.stack_depth,
                            "lr"                 : self.lr,
                            "use_cuda"           : self.use_cuda,
                            "device"             : self.device,
                            "optimizer_instance" : self.optimizer_instance}
        
        copied = StackAugmentedVAE(**kwargs_generator)
        copied.load_state_dict(self.state_dict())
        return copied

    
def time_since(since):
    s = time.time() - since
    m = s // 60
    s -= m * 60
    return '%dm %ds' % (m, s)