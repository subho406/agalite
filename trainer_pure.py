import hydra
import multiprocessing as mp
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config_pure/", config_name="default_config")
def main(config: DictConfig):
    import sys
    sys.path.append(".")
    sys.path.append("src/")
    from src_pure.purejaxrl.ppo_rnn import make_update
    from src_pure.models.agalite import BatchedAGaLiTe
    from src_pure.models.gru import BatchedGRU
    from src_pure.models.gtrxl import BatchedGTrXL
    from src_pure.purejaxrl.wrappers import FlattenObservationWrapper, LogWrapper, AutoResetEnvWrapper
    from omegaconf import DictConfig, OmegaConf
    from tqdm.contrib.logging import logging_redirect_tqdm

    import jax
    import gymnax
    
    import tqdm
    import wandb
    import logging
    

    config = OmegaConf.to_object(config)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    logger.info("Starting Job for Config:\n"+str(config))
    logger.info("Available Backends:"+str(jax.devices()))

    if config["USE_WANDB"]:
        # Initialize the wandb logger
        wandb.init(config=config,
                    project=config["WANDB_PROJECT"],
                   tags=config["WANDB_TAGS"])

    if config['RNN']['RNN_TYPE'] == "gru":
        def rnn_module():
            return BatchedGRU()

        def rnn_carry_init(batch_size):
            return BatchedGRU.initialize_carry(batch_size, config['RNN']['HIDDEN_SIZE'])
    elif config['RNN']['RNN_TYPE'] == "agalite":
        def rnn_module():
            return BatchedAGaLiTe(n_layers=config['RNN']['N_LAYERS'],
                                 d_model=config['RNN']['D_MODEL'],
                                 d_head=config['RNN']['D_HEAD'],
                                 d_ffc=config['RNN']['D_FFC'],
                                 n_heads=config['RNN']['N_HEADS'],
                                 eta=config['RNN']['ETA'],
                                 r=config['RNN']['R'])

        def rnn_carry_init(batch_size):
            return BatchedAGaLiTe.initialize_carry(batch_size, n_layers=config['RNN']['N_LAYERS'], n_heads=config['RNN']['N_HEADS'],
                                                  d_head=config['RNN']['D_HEAD'],
                                                  eta=config['RNN']['ETA'], r=config['RNN']['R'])
    elif config['RNN']['RNN_TYPE'] == "gtrxl":
        def rnn_module():
            return BatchedGTrXL(head_dim=config['RNN']['HEAD_DIM'],
                                embedding_dim=config['RNN']['EMBEDDING_DIM'],
                                head_num=config['RNN']['HEAD_NUM'],
                                mlp_num=config['RNN']['MLP_NUM'],
                                layer_num=config['RNN']['LAYER_NUM'],
                                memory_len=config['RNN']['MEMORY_LEN'],
                                )

        def rnn_carry_init(batch_size):
            return BatchedGTrXL.initialize_carry(batch_size, memory_len=config['RNN']['MEMORY_LEN'],
                                                 embedding_dim=config['RNN']['EMBEDDING_DIM'],
                                                 layer_num=config['RNN']['LAYER_NUM'])
            
    # Initialize the env
    if config["ENV_NAME"] == "craftax":
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
        env = CraftaxSymbolicEnv()
        env_params = env.default_params
        env = LogWrapper(env)
    else:
        env, env_params = gymnax.make(
            config["ENV_NAME"])
        env = FlattenObservationWrapper(env)
        env = AutoResetEnvWrapper(env)
        env = LogWrapper(env)

    rng = jax.random.PRNGKey(30)
    update_fn, runner_state = make_update(rng, config, env,
                                          env_params, rnn_module, rnn_carry_init)
    num_updates = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    global_step = 0  # Global step counter
    pbar = tqdm.tqdm(total=config['TOTAL_TIMESTEPS'])
    try:
        with logging_redirect_tqdm():
            while True:
                out = update_fn(runner_state)
                runner_state = out['runner_state']
                metric = out['metric']
                # Average the data across the episodes
                metric = jax.tree.map(
                    lambda x: (x * metric["returned_episode"]).sum()
                    / metric["returned_episode"].sum(),
                    metric
                )
                metric_float = jax.tree.map(lambda x: float(x), metric)
                logger.info("Step: %d, Metric: %s" %
                            (global_step, metric_float))
                if config["USE_WANDB"]:
                    # Convert the metric into float first before logging
                    wandb.log(metric_float, step=global_step)

                global_step += config["LOG_INTERVAL"] * \
                    config["NUM_STEPS"]*config["NUM_ENVS"]

                pbar.update(config["LOG_INTERVAL"] *
                            config["NUM_STEPS"]*config["NUM_ENVS"])
                if global_step >= config["TOTAL_TIMESTEPS"]:
                    break

    except Exception as e:
        # Log the exception using the logger
        logger.exception("An exception occurred: {}".format(e))


if __name__ == '__main__':
    mp.set_start_method('forkserver')
    main()
