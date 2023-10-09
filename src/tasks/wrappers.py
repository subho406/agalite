import gymnasium as gym

import numpy as np
import jax.numpy as jnp

from gymnasium.wrappers import AutoResetWrapper
from typing import Callable, List, Tuple, Union


#Action observation wrapper: WIP
class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Tuple((env.observation_space, gym.spaces.Box(low=0, high=1, shape=(env.action_space.n,), dtype=np.int32), gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=env.reward_range.dtype)))
        self.prev_action = None
        self.prev_reward = None

    def reset(self):
        obs = self.env.reset()
        self.prev_action = None
        self.prev_reward = None
        return (obs, self.prev_action, self.prev_reward)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.prev_action = np.zeros(self.env.action_space.n)
        self.prev_action[action] = 1
        self.prev_reward = np.array([reward])
        return (obs, self.prev_action, self.prev_reward), reward, done, info


class EpisodeStatisticsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.tick=0
        self.raw_rewards = []
    
    
    def step(self, action):
        observations, rewards, terminated, truncated, info = self.env.step(action)
        self.tick += 1
        self.raw_rewards.append(rewards)
        if terminated or truncated:
            info["episode_length"] = self.tick
            info["reward_per_episode"] = np.sum(self.raw_rewards)
            info["rewards"] = self.raw_rewards
            self.tick=0
            self.raw_rewards=[]
        return observations, rewards, terminated, truncated, info


class RecordRolloutAfterInterval(gym.Wrapper):
    def __init__(self, env,steps_interval):
        super().__init__(env)
        self.steps_count=0
        self.steps_interval=steps_interval
        self.need_to_record=False
        self.record_start=False
        self.frames=[]
    
    def step(self, action):
        if self.record_start:
            self.frames.append(self.env.render())
        observations, rewards, terminated, truncated, info = self.env.step(action)
        self.steps_count+=1
        if not self.need_to_record and not self.record_start and (self.steps_count+1)%self.steps_interval==0:
            self.need_to_record=True
        
            
        if (terminated or truncated):
            if self.need_to_record:
                self.record_start=True
                self.need_to_record=False
            elif self.record_start:
                self.record_start=False
                if len(self.frames)>0:
                    info['frames']=np.transpose(np.stack(self.frames,0).copy(),(0,3,1,2))
                self.frames=[]
        return observations, rewards, terminated, truncated, info


class RecordRollout(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.frames=[]
    
    def step(self, action):
        self.frames.append(self.env.render())
        observations, rewards, terminated, truncated, info = self.env.step(action)
        if (terminated or truncated):
            if len(self.frames)>0:
                info['frames']=np.transpose(np.stack(self.frames,0).copy(),(0,3,1,2))
            self.frames=[]
        return observations, rewards, terminated, truncated, info