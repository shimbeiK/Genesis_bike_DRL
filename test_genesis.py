import genesis as gs
import numpy as np
import torch
import time

gs.init(
    seed                = None,  # 乱数シード
    precision           = '32',  # 精度レベル ('32'/'64')
    debug               = False,  # デバッグ
    eps                 = 1e-12,  # EPS
    backend             = gs.gpu,  # バックエンドデバイス (gs.cpu/gs.cuda/gs.vulkan/gs.metal)
    theme               = 'dark',  # カラーテーマ (dark/light/dumb)
    logging_level       = None,  # ログレベル ('warning'など)
    logger_verbose_time = False,  # ログに時間情報を含めるか
)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0, 0, -9.81),
    ),
    show_viewer=True,  # either viewer or headless
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)


model = scene.add_entity(
    gs.morphs.MJCF(
        file  = 'xml/mjcf2/HBP_mjcf.xml',
        pos   = (0, 0, 0),
        euler = (0, 0, 0), # オイラー (x-y-z 規則)
        # quat  = (1.0, 0.0, 0.0, 0.0), # クォータニオン (w-x-y-z 規則)
        scale = 1.0,
    ),
)

scene.build(    
    n_envs=1, 
    env_spacing=(1.0, 1.0)
    )

# 4. 関節のインデックス（DOF Index）を取得する
# Genesisでは名前ではなく、割り当てられたID番号で関節を操作します
fork_dof = model.get_joint("tire_holder_yaw").dof_idx_local
back_dof = model.get_joint("tire_back_pitch").dof_idx_local
top_dof  = model.get_joint("tire_top_pitch").dof_idx_local

# 5. 制御モード（位置制御・力制御）のセットアップ
# フロントフォーク（tire_holder_yaw）: XMLの <position> に合わせて位置制御（PD制御）
model.set_dofs_kp([100.0], [fork_dof])
model.set_dofs_kv([1.0], [fork_dof])

# 前後のタイヤ（tire_back_pitch, tire_top_pitch）: XMLの <motor> に合わせてトルク（力）制御
# 力制御の場合は kp=0, kv=0 にしてバネ・ダンパの抵抗をなくします
model.set_dofs_kp([0.0, 0.0], [back_dof, top_dof])
model.set_dofs_kv([0.0, 0.0], [back_dof, top_dof])

# 6. シミュレーションループ
print("シミュレーションを開始します。ウィンドウを閉じるにはターミナルで Ctrl+C を押してください。")

target_torque_back = 0.01
target_torque_top  = 0.01
for i in range(400): #n step * dt sec monitoring
# while True:
    model.control_dofs_position(np.array([np.deg2rad(45)]), [fork_dof])
    # 例：前後のタイヤに 2.0 Nm のトルクをかけ続ける
    model.control_dofs_force(np.array([target_torque_back, target_torque_top]), [back_dof, top_dof])
    # ==== 状態の取得は毎ステップ、ループの中で行う ====
    dof_vels = model.get_dofs_velocity()
    fork_vel = dof_vels[:, fork_dof]
    back_vel = dof_vels[:, back_dof]
    top_vel  = dof_vels[:, top_dof]
    print(f"Step {i}: Fork Vel = {fork_vel}, Back Vel = {back_vel}, Top Vel = {top_vel}")

    base_ang_vel = model.get_ang()
    print(f"Step {i}: Base Angular Vel = {base_ang_vel}")
    
    base_quat = model.get_quat()
    # ロボットの姿勢(w, x, y, z)を展開し、直接 Roll と Pitch を計算
    w, x, y, z = model.get_quat().T
    roll = torch.atan2(2 * (w*x + y*z), 1 - 2 * (x**2 + y**2))
    # pitch = torch.asin((2 * (w*y - z*x)).clip(-1.0, 1.0))
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y**2 + z**2)) # 必要ならコメントアウト解除

    print(f"ステップ {i} | Roll: {torch.rad2deg(roll[0]):.1f}度 | Yaw: {torch.rad2deg(yaw[0]):.1f}度")
    scene.step()
    # time.sleep(0.1)
# scene.stop_recording()