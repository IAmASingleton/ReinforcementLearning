# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:49:54 2024

@author: Patrick
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f # euda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super (Actor, self).__init__()
        
        self.input_layer = nn.Linear(state_dim, 256)
        self.hidden_layer = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        a = f.relu(self.input_layer(state))
        a = f.relu(self.hidden_layer(a))
        a = self.max_action * torch.tanh(self.output_layer(a))
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.input_layer_1 = nn.Linear(state_dim + action_dim, 256)
        self.hidden_layer_1 = nn.Linear(256, 256)
        self.output_layer_1 = nn.Linear(256, 1)
        
        # Q2 architecture
        self.input_layer_2 = nn.Linear(state_dim + action_dim, 256)
        self.hidden_layer_2 = nn.Linear(256, 256)
        self.output_layer_2 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action],1)
        
        q1 = f.relu(self.input_layer_1(sa))
        q1 = f.relu(self.hidden_layer_1(q1))
        q1  = self.output_layer_1(q1)
        
        q2 = f.relu(self.input_layer_2(sa))
        q2 = f.relu(self.hidden_layer_2(q2))
        q2 = self.output_layer_2(q2)
        
        return q1,q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action],1)
        
        q1 = f.relu(self.input_layer_1(sa))
        q1 = f.relu(self.hidden_layer_1(q1))
        q1  = self.output_layer_1(q1)
        return q1

class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
            ):
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.depcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.iterations_total = 0
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=256):
        self.iterations_total += 1
        
        state, action, next_state, reward, running = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                    self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + running * self.discount * target_Q
            
        current_Q1, current_Q2 = self.critic(state, action)
        
        critic_loss = f.mse_loss(current_Q1, target_Q) + f.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        if self.total_it % self.policy_freq == 0:
            actor_loss = self.critic.Q1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
    
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        
    def load(self, filename):
        self.critic.load_tate_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor.load_tate_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actorc_target = copy.deepcopy(self.actor)