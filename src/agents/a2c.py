import jax
import jax.numpy as jnp
import optax
import rlax
import tqdm

from src.models.actor_critic import *
from typing import Callable
from src.agents.base_agent import ActorCriticModel,BaseAgent


class A2CAgent(BaseAgent):

    def __init__(self,train_envs,eval_env,rollout_len,
                        repr_model_fn:Callable,seq_model_fn:Callable,
                        actor_fn:Callable,critic_fn:Callable,optimizer:optax.GradientTransformation,gamma:float=0.99,lamb:float=0.95,
                        entropy_coef=0.01, value_loss_coef=0.5,arg_max=False) -> None:
        super(A2CAgent,self).__init__(train_envs=train_envs,eval_env=eval_env,rollout_len=rollout_len,repr_model_fn=repr_model_fn,seq_model_fn=seq_model_fn,
                        actor_fn=actor_fn,critic_fn=critic_fn)
        self.num_actors=self.env.num_envs
        self.gamma=gamma
        self.lamb=lamb
        self.entropy_coef=entropy_coef
        self.value_loss_coef=value_loss_coef
        self.optimizer=optimizer
        self.arg_max=arg_max
        
        @jax.jit
        def loss_fn(params,random_key,observations,actions,rewards,terminations,h_tickminus1):
            Glambda_fn=jax.vmap(rlax.lambda_returns)
            entropy_fn=jax.vmap(rlax.entropy_loss)
            pg_fn=jax.vmap(rlax.policy_gradient_loss)
            gammas=self.gamma*(1-terminations)
            lambdas=self.lamb*jnp.ones(self.num_actors)
            #Cacluate the values and log likelihoods for the timesteps {tick} - {tick+rollout_len+1}
            logits_diff,values_diff,_=self.actor_critic_fn(random_key,params,observations,terminations,h_tickminus1)
            #Calculate Lamba for timesteps G_{tick} - G_{tick+rollout_len} using 
            #rewards, gammas, lambdas values at timesteps {tick+1} - {tick+rollout_len+1}
            Glambdas=Glambda_fn(rewards[:,1:],gammas[:,1:],
                              values_diff[:,1:],lambdas)
            #Calculate the critic loss using timesteps {tick} - {tick+rollout_len}
            advantages=(jax.lax.stop_gradient(Glambdas)-values_diff[:,:-1])
            value_loss=(advantages**2).mean()
            #Calculate the actor loss using timesteps {tick} - {tick+rollout_len}
            logits_diff=logits_diff[:,:-1]
            actor_loss=pg_fn(logits_diff,actions,jax.lax.stop_gradient(advantages),
                             jnp.ones((self.num_actors,self.rollout_len))).mean()
            #Calculate the entropy
            entropy_loss=entropy_fn(logits_diff,jnp.ones((self.num_actors,self.rollout_len))).mean()
            
            loss=actor_loss+self.entropy_coef*entropy_loss+self.value_loss_coef*value_loss
            return loss,(value_loss,entropy_loss,actor_loss)       
        self.loss_fn=loss_fn
        
        @jax.jit
        def step_fn(params,optimizer_state,random_key,observations,actions,rewards,terminations,h_tickminus1):
                grad_fn=jax.value_and_grad(loss_fn,has_aux=True)
                (loss,(value_loss,entropy_loss,actor_loss)),grads=grad_fn(params,random_key,observations,actions,rewards,
                                                         terminations,h_tickminus1)
                updates,optimizer_state=self.optimizer.update(grads,optimizer_state)
                params=optax.apply_updates(params,updates)
                return (loss,(value_loss,entropy_loss,actor_loss)),params,optimizer_state 
        self.step_fn=step_fn
        

    def reset(self,params_key,random_key):
        #Reset the Agent and initilize the parameters
        super(A2CAgent,self).reset(params_key,random_key)
        self.optimizer_state=self.optimizer.init(self.params)
    
    
    def step(self,random_key):
        #Unroll actor for rollout_len steps
        h_tickminus1=jax.tree_map(lambda x: x,self.h_tickminus1) #Copy the hidden state
        #Need to remove values
        unroll_key,update_key=jax.random.split(random_key)
        unroll_data=self.unroll_actors(unroll_key)
        observations,actions,rewards,terminations,infos=unroll_data.observations,unroll_data.actions, \
                                            unroll_data.rewards,unroll_data.terminations,unroll_data.infos
        
        #Take a learner step
        (loss,(value_loss,entropy_loss,actor_loss)),self.params,self.optimizer_state=self.step_fn(self.params,
                                self.optimizer_state,update_key,observations,actions,
                                rewards,terminations,h_tickminus1)
        return (loss,(value_loss,entropy_loss,actor_loss,rewards[:,1:]),infos)

    

            
