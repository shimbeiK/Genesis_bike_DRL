import genesis as gs
import math

class PythonConfig:
    def get_train_cfg(exp_name, max_iterations):
        train_cfg_dict = {
            "algorithm": {
                "class_name": "PPO",
                "clip_param": 0.2,
                "desired_kl": 0.01,
                "entropy_coef": 0.01,
                "gamma": 0.99,
                "lam": 0.95,
                "learning_rate": 0.001, # Can be tweaked to match parameters_ppo.json
                "max_grad_norm": 1.0,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "schedule": "adaptive",
                "use_clipped_value_loss": True,
                "value_loss_coef": 1.0,
            },
            "init_member_classes": {},
            "policy": {
                "class_name": "ActorCritic",
                "activation": "elu",
                "actor_hidden_dims": [256, 128, 64], # Slightly smaller for simpler 3D observation
                "critic_hidden_dims": [256, 128, 64],
                "init_noise_std": 1.0,
            },
            "runner": {
                "checkpoint": -1,
                "experiment_name": exp_name,
                "load_run": -1,
                "log_interval": 1,
                "max_iterations": max_iterations,
                "record_interval": -1,
                "resume": False,
                "resume_path": None,
                "run_name": "bike_balancing",
            },
            "runner_class_name": "OnPolicyRunner",
            "num_steps_per_env": 24,
            "save_interval": 100,
            "empirical_normalization": None,
            "seed": 1,
        }
        return train_cfg_dict
    
    def get_cfgs():
        env_cfg = {
            "num_actions": 2, # Only 1 action: Torque applied to the balancing wheel
            
            # Update this to match the specific joint name in your Genesis URDF/XML
            "joint_names": ["tire_holder_yaw", "tire_back_pitch"], 
            "default_joint_angles": {  
                "tire_holder_yaw": math.radians(-60),
                "tire_back_pitch": 0.0,
            },
            
            # 前輪（ステアリング）用の位置制御ゲイン
            # ※値はモデルの重さ等に合わせて調整（チューニング）が必要です
            "steering_kp": 100.0, 
            "steering_kd": 1.0,  
            
            # 後輪用のトルク制御ゲイン（トルク制御なのでゼロ）
            "drive_kp": 0.0, 
            "drive_kd": 0.0,  
            
            # Termination bounds (converted np.pi/6 to approx 30 degrees)
            "termination_if_roll_greater_than": math.radians(30.0),
            # "termination_if_pitch_greater_than": 180.0, # Ignored for balancing task
            
            # Initial Base state
            "base_init_pos": [0.0, 0.0, 0.03], # Adjust height based on your bike model
            "base_init_quat": [1.0, 0.0, 0.0, 0.0],
            "initial_tilt_deg": math.radians(4), # The 3.7 degree initialization from MuJoCo code
            
            "episode_length_s": 40.0, 
            "resampling_time_s": 4.0, 

            # Action scale (ネットワーク出力 [-1, 1] をそれぞれの物理量に変換)
            "steering_angle_scale": math.radians(60), # Action 0 のスケール（角度）
            "drive_torque_scale": 0.021,              # Action 1 のスケール（トルク）            
            "simulate_action_latency": False, # Turned off for simpler dynamics matching MuJoCo
            "clip_actions": 1.0, 
        }
        
        obs_cfg = {
            "num_obs": 7, # Roll angle (rad), Angular Velocity (rad/s), Angular Acceleration (rad/s^2)
            "obs_scales": {
                "roll": 1.0,
                "ang_vel": 1.0,
                "ang_acc": 1.0,
            },
        }
        
        reward_cfg = {
            "reward_scales": {
                "upright_posture": 5.0,     # Matches: 3*(45deg - abs(imu))/45deg
                "angular_vel_penalty": -0, # Matches: -0.09 * abs(angular_vel)
                "survival_bonus": 2.0,      # Matches the step_count / 100.0 logic in MuJoCo
            },
        }
        
        # Commands are not strictly needed for stationary balancing, but kept to prevent pipeline breakage
        command_cfg = {
            "num_commands": 0, 
        }

        return env_cfg, obs_cfg, reward_cfg, command_cfg