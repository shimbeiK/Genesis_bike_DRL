import math
import torch
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower

class StandingEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", False)
        self.dt = 0.01 # 100 Hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # --- Scene Setup ---
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=5, # Matches MuJoCo's frame_skip=5
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, -1.5, 1.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # IMPORTANT: Ensure your XML here points to the bike model
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/mjcf2/HBP_mjcf.xml", # Update this path if necessary
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )

        self.scene.build(n_envs=num_envs)

        ## 前輪（ステアリング）と後輪のインデックスを別々に取得
        self.steering_dof_idx = torch.tensor(
            [self.robot.get_joint("tire_holder_yaw").dof_start],
            dtype=gs.tc_int, device=gs.device,
        )
        self.drive_dof_idx = torch.tensor(
            [self.robot.get_joint("tire_back_pitch").dof_start],
            dtype=gs.tc_int, device=gs.device,
        )
        
        # 前輪には位置制御用のゲインをセット（リストの要素は1つだけ）
        self.robot.set_dofs_kp([self.env_cfg["steering_kp"]], self.steering_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["steering_kd"]], self.steering_dof_idx)

        # 後輪にはトルク制御用のゲインをセット（リストの要素は1つだけ）
        self.robot.set_dofs_kp([self.env_cfg["drive_kp"]], self.drive_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["drive_kd"]], self.drive_dof_idx)
        
        # --- Buffers ---
        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:] if joint.name in self.env_cfg["default_joint_angles"]],
            dtype=gs.tc_float, device=gs.device,
        )

        # ベース(7要素: 位置3 + クォータニオン4)を除いた、純粋な関節数分の初期角度を用意する
        num_joints = self.robot.n_qs - 7
        dof_pos_list = []
        
        # 環境の全関節(0番目はルートなので1番目から)をループして初期値を取得
        for joint in self.robot.joints[1:]:
            if joint.name in self.env_cfg["default_joint_angles"]:
                dof_pos_list.append(self.env_cfg["default_joint_angles"][joint.name])
            else:
                dof_pos_list.append(0.0) # 設定がない関節は0にする
                
        # もし取得した関節数が実際の自由度と合わない場合の安全対策
        while len(dof_pos_list) < num_joints:
            dof_pos_list.append(0.0)
            
        self.init_dof_pos = torch.tensor(dof_pos_list[:num_joints], dtype=gs.tc_float, device=gs.device)
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
                
        # GPU Buffers
        self.obs_buf = torch.empty((self.num_envs, self.num_obs), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        
        # For Observation (Roll, Ang Vel, Ang Accel)
        self.base_euler = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.last_base_ang_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_accel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)

        self.extras = dict()  
        self.extras["observations"] = dict()

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt 
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        
        # Action 0: 前輪のステアリング（位置制御）
        steering_target = self.actions[:, 0:1] * self.env_cfg["steering_angle_scale"]
        self.robot.control_dofs_position(steering_target, self.steering_dof_idx)
        
        # Action 1: 後輪の駆動（トルク制御）
        drive_torque = self.actions[:, 1:2] * self.env_cfg["drive_torque_scale"]
        self.robot.control_dofs_force(drive_torque, self.drive_dof_idx)
        
        self.scene.step()

        # 2. Update Kinematics
        self.episode_length_buf += 1 
        base_quat = self.robot.get_quat()
        inv_base_quat = inv_quat(base_quat)
        
        # Get Roll, Pitch, Yaw
        self.base_euler = quat_to_xyz(transform_quat_by_quat(self.inv_base_init_quat, base_quat), rpy=True, degrees=False)
        
        # Update Angular Velocities and Accelerations
        current_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.base_ang_accel = (current_ang_vel - self.last_base_ang_vel) / self.dt
        self.base_ang_vel = current_ang_vel
        self.last_base_ang_vel.copy_(current_ang_vel)

        # 3. Compute Rewards
        self.rew_buf.zero_() 
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew 
            self.episode_sums[name] += rew

        # 4. Termination Logic (Angle > 30 deg)
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        # Checking Roll (X-axis) 
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        self._reset_idx(self.reset_buf)
        self._update_observation()

        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    # reset envinonment(s). can reset all envs (envs_idx=None) or a subset of envs (envs_idx is a boolean mask)
    def _reset_idx(self, envs_idx=None):
        # NOTE: If you want to spawn the bike with the 3.7 degree tilt, it can be applied to `self.init_qpos` here

        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
            self.base_euler.zero_() # ← 追加
            self.base_ang_vel.zero_()
            self.last_base_ang_vel.zero_()
            self.actions.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            self.base_euler.masked_fill_(envs_idx[:, None], 0.0) # ← 追加
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.last_base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            value.masked_fill_(envs_idx, 0.0)

    # define observation enviroment
    def _update_observation(self):
        # [Roll, Angular Velocity (X), Angular Acceleration (X)]
        self.obs_buf = torch.concatenate(
            (
                self.base_euler[:, 0:1] * self.obs_scales["roll"], 
                self.base_ang_vel[:, 0:1] * self.obs_scales["ang_vel"], 
                self.base_ang_accel[:, 0:1] * self.obs_scales["ang_acc"],
            ),
            dim=-1,
        )

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.obs_buf, None

    # ================================================
    # Reward Functions translated from MuJoCo
    # ================================================
    def _reward_upright_posture(self):
        # reward += 3*(np.deg2rad(45) - abs(imu)) / np.deg2rad(45)
        roll = torch.abs(self.base_euler[:, 0])
        target_rad = 45.0 * math.pi / 180.0
        return (target_rad - roll) / target_rad

    def _reward_angular_vel_penalty(self):
        # reward -= 0.09 * abs(angular_vel)
        return torch.abs(self.base_ang_vel[:, 0])

    # def _reward_survival_bonus(self):
    #     # return self.step_count / 100.0 (Translated to a steady positive stream in PPO)
    #     return self.episode_length_buf / 100.0