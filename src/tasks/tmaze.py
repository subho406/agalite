import gymnasium as gym
import gymnasium.spaces as spaces
import gymnasium.wrappers as wrappers
import numpy as np





class BakkersTMazeWDistractors(gym.Env):
    
    def __init__(self,corridor_len,num_distractors,render_mode=None,seed=None):
        """Starting: up: 011, down:110
            corridor: 101
            Tjunction: 010
        """
        self.corridor_len=corridor_len
        self.num_distractors=num_distractors
        self.render_mode=render_mode
        self.observation_space=spaces.MultiBinary(3+self.num_distractors)
        self.action_space=spaces.Discrete(4) #0:up,1:down,2:left,3:right
        self.rng=np.random.default_rng(seed)
        self.up_cue=np.array([0,1,1])
        self.down_cue=np.array([1,1,0])
        self.corridor_cue=np.array([1,0,1])
        self.tjunction_cue=np.array([0,1,0])
        self.last_success=False
        


    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng=np.random.default_rng(seed)
        self.current_pos=0
        self.initial_obs=self.rng.choice([0,1])
        if self.initial_obs==0:
            current_obs=np.concatenate([self.up_cue,np.zeros(self.num_distractors)]) #First observation does not have distractor bits
            self.current_cue=0 #0:up,1:down
        else:
            current_obs=np.concatenate([self.down_cue,np.zeros(self.num_distractors)])
            self.current_cue=1
        info={"success": self.last_success,"new_cue":str(self.current_cue)}
        self.last_success=False #Reset success
        return current_obs,info
    
    def _get_obs(self):
        if self.current_pos==self.corridor_len-1:
            #Tjunction 
            obs=np.concatenate([self.tjunction_cue,self.rng.choice([0,1],self.num_distractors)])
        else:
            #Corridor
            obs=np.concatenate([self.corridor_cue,self.rng.choice([0,1],self.num_distractors)])
        return obs

    def step(self,action):
        if self.current_pos==self.corridor_len-1:
            if action==0 or action==1:
                if action==self.current_cue:
                    reward=4
                    self.last_success=True
                else:
                    reward=-1
                    self.last_success=False
                terminated=True
                info={}
            else:
                reward=-0.1
                success=False
                terminated=False
                info={}
            obs=self._get_obs()
            truncated=False
            
        else:
            if action==2:
                #left
                self.current_pos=max(self.current_pos-1,0)
            elif action==3:
                #right
                self.current_pos=min(self.current_pos+1,self.corridor_len-1)
            obs=self._get_obs()
            reward=-0.1
            terminated=False
            truncated=False
            info={}
        return obs,reward,terminated,truncated,info



class SimplerTMazeWDistractors(gym.Env):
    
    def __init__(self,corridor_len,num_distractors,render_mode=None,seed=None):
        """Starting: up: 011, down:110
            corridor: 101
            Tjunction: 010
        """
        self.corridor_len=corridor_len
        self.num_distractors=num_distractors
        self.render_mode=render_mode
        self.observation_space=spaces.Box(0.0,1.0,(10+self.num_distractors,))
        self.action_space=spaces.Discrete(4) #0:up,1:down,2:left,3:right
        self.rng=np.random.default_rng(seed)
        self.up_cue=np.array([0,1.])
        self.down_cue=np.array([1.0,0])
        self.no_cue=np.array([0,0.])
        self.last_success=False
        
    def _get_position(self):
        """Converts a decimal number to its corresponding Gray code binary representation as a numpy array."""
        binary = np.binary_repr(self.current_pos, width=8)
        gray = np.zeros(8, dtype=float)
        gray[0] = int(binary[0])
        gray[1:] = np.bitwise_xor(np.array([int(bit) for bit in binary[:-1]]), np.array([int(bit) for bit in binary[1:]]))
        return gray

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng=np.random.default_rng(seed)
        self.current_pos=0
        self.initial_obs=self.rng.choice([0,1])
        if self.initial_obs==0:
            current_obs=np.concatenate([self.up_cue,self._get_position(),np.zeros(self.num_distractors)]) #First observation does not have distractor bits
            self.current_cue=0 #0:up,1:down
        else:
            current_obs=np.concatenate([self.down_cue,self._get_position(),np.zeros(self.num_distractors)])
            self.current_cue=1
        info={"success": self.last_success,"new_cue":self.current_cue}
        
        self.last_success=False #Reset success
        return current_obs,info
    
    def _get_obs(self):
        if self.current_pos==self.corridor_len-1:
            #Tjunction 
            obs=np.concatenate([self.no_cue,self._get_position(),self.rng.choice([0,1],self.num_distractors)])
        else:
            #Corridor
            obs=np.concatenate([self.no_cue,self._get_position(),self.rng.choice([0,1],self.num_distractors)])
        return obs

    def step(self,action):
        if self.current_pos==self.corridor_len-1:
            if action==0 or action==1:
                if action==self.current_cue:
                    reward=4
                    self.last_success=True
                else:
                    reward=-1
                    self.last_success=False
                terminated=True
                info={}
                info['episode_extra_stats']={"success": int(self.last_success),"new_cue":self.current_cue}
            else:
                reward=-0.1
                success=False
                terminated=False
                info={}
            obs=self._get_obs()
            truncated=False
            
        else:
            if action==2:
                #left
                self.current_pos=max(self.current_pos-1,0)
            elif action==3:
                #right
                self.current_pos=min(self.current_pos+1,self.corridor_len-1)
            obs=self._get_obs()
            reward=-0.1
            terminated=False
            truncated=False
            info={}
        
        return obs,reward,terminated,truncated,info
    
def create_tmazev1(**kwargs):
    env=wrappers.TimeLimit(BakkersTMazeWDistractors(corridor_len=kwargs.get("corridor_len",10),num_distractors=kwargs.get("num_distractors",6),
                    render_mode=kwargs.get("render_mode",None),seed=kwargs.get("seed",None)),max_episode_steps=kwargs.get("max_episode_steps",5000))
    return env

def create_tmazev2(**kwargs):
    env=wrappers.TimeLimit(SimplerTMazeWDistractors(corridor_len=kwargs.get("corridor_len",10),num_distractors=kwargs.get("num_distractors",6),
                    render_mode=kwargs.get("render_mode",None),seed=kwargs.get("seed",None)),max_episode_steps=kwargs.get("max_episode_steps",1000))
    return env