from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass 
class TaskDB2PiperLidarPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    check_for_nan = True
    experiment_name = "task_d_b2piper_lidar_teacher"

    obs_groups = {
        "policy": ["proprio", "extero"],
        "critic": ["proprio", "extero"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu"
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,                     
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0
    )
