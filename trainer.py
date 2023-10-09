import hydra
import multiprocessing as mp
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="default_config")
def main(config: DictConfig):
    from cmath import isnan
    import sys
    sys.path.append('./')
    sys.path.append('../')
    import numpy as np
    import random
    import json
    import argparse
    import wandb
    import jax
    import jax.numpy as jnp
    import itertools
    from src.trainers.trainers_control import ControlTrainer
    import pandas as pd
    from tqdm import tqdm
    import numpy as np
    
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import Pool 
    import multiprocessing as mp
    from tqdm.contrib.logging import logging_redirect_tqdm
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    task_to_trainer={
        'minigrid_pixel':ControlTrainer,
        'minigrid_onehot':ControlTrainer,
        'tmazev1':ControlTrainer,
        'tmazev2':ControlTrainer,
        'memory_gym':ControlTrainer,
        'popgym':ControlTrainer,
        'memory_maze':ControlTrainer,
        'procgen':ControlTrainer,
    }


    logger.info("Starting Job for Config:\n"+str(OmegaConf.to_yaml(config)))
    logger.info("Available Backends:"+str(jax.devices()))
    tags=config.tags.split(',') if config.tags is not None else []
    if config.use_wandb:
        run = wandb.init(project=config.project_name,tags=tags,settings=wandb.Settings(start_method="fork"),config=OmegaConf.to_container(config))
    else:
        run=None
    key=jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    trainer_config=config.trainer
    env_config=config.task
   
    #Train the model
    kwargs={'global_args':config,'trainer_config':trainer_config,'env_config':env_config,
            'seed':config.seed,'key':key,'wandb_run':run}
    trainer=task_to_trainer[env_config['task']](**kwargs)
    pbar = tqdm(total=config.steps)
    step_count=0
    last_step_count=0
    try:
        with logging_redirect_tqdm():
            while True:
                loss,metrics,step_count=trainer.step()
                pbar.update(n=step_count-last_step_count)
                last_step_count=step_count
                if metrics is not None:
                    logger.info("Seed: "+str(config.seed)+" Steps: "+str(step_count)+" Metrics: "+str(metrics))
                    if config.use_wandb: run.log({'seed':config.seed,**metrics
                                                    },step=step_count)


                if step_count>=config.steps:
                    break
    except Exception as e:
        # Log the exception using the logger
        logger.exception("An exception occurred: {}".format(e))
    pbar.close()
    if config.use_wandb:
        wandb.finish()
        #Need to do something about logging val_metric
    exit()


if __name__=='__main__':
    mp.set_start_method('forkserver')
    main()

    