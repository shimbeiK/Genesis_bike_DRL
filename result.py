import argparse
import os, math
import pickle
from importlib import metadata

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner # type: ignore

import genesis as gs

# from bike_env_NonSteer_withGemini import StandingEnv
from bike_env_NonSteer import StandingEnv
from rslrl_cfg_bike_standing_nonSteering import PythonConfig

file_num = 400
evv_cfg, _, _, _ = PythonConfig.get_cfgs()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="bike-standing-better")
    parser.add_argument("--ckpt", type=int, default=file_num)
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = StandingEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    counter = 0
    with torch.no_grad():
        while True:
            counter += 1
            print("counter:", counter)
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)
            # print(torch.clip(actions, -env_cfg["clip_actions"], env_cfg["clip_actions"]))
            print(math.degrees(obs[0][0]))
            if(dones):
                print("ouch!")
                counter = 0


if __name__ == "__main__":
    main()