import json
import numpy as np
import jax.numpy as jnp
import gymnasium as gym
import minigrid

from abc import ABCMeta,abstractmethod,abstractproperty
from src.tasks.nextcharprediction import *
from src.tasks.envs.tracepatterning import *
from src.tasks.envs.wrappers import *

class BaseDataLoader(metaclass=ABCMeta):
    @abstractmethod
    def step(self):
        """Takes one timestep and returns a data sample/batch
        """
        pass

    @abstractmethod
    def init(self,args):
        """Inits dataset with config
        """
        pass


    def load_config_file(self,config_path):
        with open(config_path) as f:
            config=json.loads(f.read())
        self.config=config
    
    def get_config(self):
        return self.config


class NextCharDataLoader(BaseDataLoader):
    
    def init(self,seed,config):
        self.config=config
        if self.config['task']=='anbn':
            self.env=AnBn(self.config['k'],self.config['l'])
            self.input=self.env.step()
        elif self.config['task']=='DistantBrackets':
            self.env=DistantBrackets(self.config['s'],self.config['k'],
                                        self.config['a'])
            self.input=self.env.step()
    
    def step(self):
        if self.config['task']=='anbn' or self.config['task']=='DistantBrackets':
            target=self.env.step()
            input=self.input
            self.input=target
            return input,target
    
    @property
    def input_size(self):
        if self.config['task']=='anbn' or self.config['task']=='DistantBrackets':
            return self.env.input_size


class AnimalBehaviourDataLoader(BaseDataLoader):
    def init(self,seed,config):
        self.config=config
        if self.config['task']=='trace_patterning':
            self.env=TracePatterning(seed=seed,ISI_interval=self.config['ISI_interval'],
                                    ITI_interval=self.config['ITI_interval'],gamma=self.config['gamma'],
                                    num_CS=self.config['num_CS'],num_activation_patterns=self.config['num_activation_patterns'],
                                    activation_patterns_prob=self.config['activation_patterns_prob'],
                                    num_distractors=self.config['num_distractors'],activation_lengths=self.config['activation_lengths'],
                                    noise=self.config['noise'])
            self.env.reset()
            self.O_t=self.env.step(None).observation
        
    def step(self):
        if self.config['task']=='trace_patterning':
            new_step=self.env.step(None)
            O_t=self.O_t.copy()
            O_tplus1=new_step.observation
            R_tplus1=jnp.array(new_step.reward,dtype=jnp.float32)
            self.O_t=O_tplus1.copy()
            return O_t,O_tplus1,R_tplus1


class MiniGridDataLoader(BaseDataLoader):
    """
    Dataloader for MiniGrid environments, takes actions specified as indices"""
    def init(self, seed,config,num_actors):
        self.config=config
        self.num_actors=num_actors
        def make_env(task):
            def thurn():
                env=EpisodeStatisticsWrapper(minigrid.wrappers.ImgObsWrapper(minigrid.wrappers.OneHotPartialObsWrapper(
                                    minigrid.wrappers.ViewSizeWrapper(gym.make(task,
                                                        max_episode_steps=self.config['max_steps'],autoreset=True,render_mode='rgb_array'),
                                                                        self.config['view_size']))))
                return env
            return thurn
        self.env=[make_env(config['env']) for i in range(self.num_actors)]
        self.env=gym.vector.SyncVectorEnv(self.env)
        #Such that each environement is seeded differently
        self.env.reset(seed=[seed+i for i in range(self.num_actors)])
        #Create single agent env for evaluation
        self.single_agent_env=RecordRollout(make_env(config['env'])())
        self.seed=seed

    @property
    def num_envs(self):
        return self.num_actors


    def reset(self):
        """_summary_

        Returns:
            _type_: observations
        """
        return self.env.reset(seed=[self.seed+i for i in range(self.num_actors)])[0]
    
    def action_space(self):
        return self.env.action_space[0].n

    def step(self,actions):
        observations, rewards, terminated, truncated,info=self.env.step(actions)
        terminated=jnp.logical_or(terminated,truncated)
        return jnp.array(observations),jnp.array(rewards),jnp.array(terminated,dtype=jnp.int8),info
    
    def reset_single_agent(self):
        #Seed of this environment is different from the vectorized one
        return self.single_agent_env.reset(seed=self.seed+self.num_actors)[0]

    def step_single_agent(self,action):
        observation, reward, terminated, truncated,info=self.single_agent_env.step(action)
        terminated=jnp.logical_or(terminated,truncated)
        return jnp.array(observation),jnp.array(reward),jnp.array(terminated,dtype=jnp.int8),info