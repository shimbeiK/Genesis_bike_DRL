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
        # Action is now only drive torque (1 dimension)
        self.num_actions = 1 
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", False)
        self.dt = env_cfg["dt"] # 100 Hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        
        # We only need to track the last drive torque now
        self.last_drive_torque = torch.zeros((self.num_envs, 1), dtype=gs.tc_float, device=gs.device)

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
        self.steering_dof_idx = self.robot.get_joint("tire_holder_yaw").dofs_idx_local
        self.drive_dof_idx = self.robot.get_joint("tire_back_pitch").dofs_idx_local
        
        # 前輪には位置制御用のゲインをセット
        self.robot.set_dofs_kp([self.env_cfg["steering_kp"]], dofs_idx_local=self.steering_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["steering_kd"]], dofs_idx_local=self.steering_dof_idx)

        # 後輪にはトルク制御用のゲインをセット
        self.robot.set_dofs_kp([self.env_cfg["drive_kp"]], dofs_idx_local=self.drive_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["drive_kd"]], dofs_idx_local=self.drive_dof_idx)
        
        # --- Buffers ---
        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        
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
        
        # Action buffer is now size [num_envs, 1]
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        
        # For Observation (Roll, Ang Vel, Ang Accel)
        self.base_euler = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.last_base_ang_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_accel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.drive_vel = torch.zeros((self.num_envs, 1), dtype=gs.tc_float, device=gs.device)
        self.total_odometry = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.extras = dict()  
        self.extras["observations"] = dict()
        self.reward_functions, self.episode_sums = dict(), dict()
        self.reward_scales = dict(reward_cfg["reward_scales"])
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt 
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
            
        # Hardcode steering target to -45 degrees (matching default_joint_angles)
        self.steering_target = torch.full((self.num_envs, 1), self.env_cfg["default_joint_angles"]["tire_holder_yaw"], dtype=gs.tc_float, device=gs.device)
        # self.drive_torque = self.env_cfg["default_joint_angles"]["tire_back_pitch"] * self.env_cfg["drive_torque_scale"]
        # self.robot.control_dofs_force(self.drive_torque, dofs_idx_local=self.drive_dof_idx)
        # self.scene.step()

    def step(self, actions):
        # _ = self._reward_odometry_penalty()
        # print(actions.cpu().numpy())  # Debug print for actions
        # Clip actions (only drive torque now)
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        
        # Apply fixed steering angle
        self.robot.control_dofs_position(self.steering_target, dofs_idx_local=self.steering_dof_idx)
        # print(self.steering_target.squeeze().cpu().numpy())  # Debug print for steering target
        
        # Action 0: Drive torque (previously Action 1)
        self.drive_torque = self.actions[:, 0:1] * self.env_cfg["drive_torque_scale"]
        self.robot.control_dofs_force(self.drive_torque, dofs_idx_local=self.drive_dof_idx)
        
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

        self.drive_vel = self.robot.get_dofs_velocity(self.drive_dof_idx)

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
        # 現在のロボットのベース位置を取得 (形状: [num_envs, 3])
        base_pos = self.robot.get_pos()
        # self.reset_buf |= torch.abs(self.total_odometry) > self.env_cfg["termination_if_odometry_greater_than"]
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
            self.base_euler.zero_() 
            self.base_ang_vel.zero_()
            self.last_base_ang_vel.zero_()
            self.actions.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
            self.drive_vel.zero_()
            self.last_drive_torque.zero_()
            self.total_odometry.zero_()
        else:
            self.base_euler.masked_fill_(envs_idx[:, None], 0.0) 
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.last_base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)
            self.drive_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.last_drive_torque.masked_fill_(envs_idx[:, None], 0.0)
            self.total_odometry.masked_fill_(envs_idx, 0.0)

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
                self.drive_vel,            # 後輪の回転速度
                self.actions               # 前回AIが出力したアクション（トルク指令値）
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
        roll = torch.abs(self.base_euler[:, 0])
        target_rad = 45.0 * math.pi / 180.0
        return (target_rad - roll) / target_rad

    def _reward_angular_vel_penalty(self):
        return (1 - torch.minimum(torch.abs(self.base_ang_vel[:, 0]) / 1.0, 
                             torch.tensor(1.0, device=self.base_ang_vel.device)))
    
    def _reward_torque_change_penalty(self):
        reward =  torch.abs(self.drive_torque - self.last_drive_torque) / 2.0
        self.last_drive_torque.copy_(self.drive_torque)
        return reward.squeeze(-1)

    def _reward_odometry_penalty(self):
        delta_odometory = 3.1 * self.dt * self.drive_vel.squeeze(-1)
        reward = torch.abs(self.total_odometry) - torch.abs(self.total_odometry + delta_odometory)
        self.total_odometry += delta_odometory
        # print(reward.cpu().numpy())  # Debug print for odometry penalty
        # print(self.total_odometry.cpu().numpy())  # Debug print for odometry penalty
        return reward.squeeze(-1)

    def _reward_survival_bonus(self):
        return 1.0