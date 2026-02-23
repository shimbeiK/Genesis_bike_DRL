import argparse  
import os       
import pickle 
import shutil    
from importlib import metadata 
from rslrl_cfg_bike_standing import PythonConfig  
import genesis as gs 
from bike_env import StandingEnv 
# --- rsl_rlパッケージの厳密なバージョンチェック ---
# Genesisは純正のrsl-rlではなく、軽量化されたフォーク版(rsl-rl-lib 2.2.4)を要求するため、ここでチェックを行う
try:
    try:
        if metadata.version("rsl-rl"): 
            raise ImportError 
    except metadata.PackageNotFoundError: 
        if metadata.version("rsl-rl-lib") != "2.2.4": 
            raise ImportError 
except (metadata.PackageNotFoundError, ImportError) as e:
    # 最終的に条件を満たさない場合、正しいインストールコマンドを提示してプログラムを強制終了させる
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner # type: ignore

num_envs = 4096
num_iterations = 2501
# --- メイン処理 ---
def main():
    parser = argparse.ArgumentParser() # 引数解析器の初期化

    # 実験名（ログフォルダ名に使われる）。
    parser.add_argument("-e", "--exp_name", type=str, default="bike-standing")

    # 並列で動かす環境の数。デフォルトは4096個（GPUメモリに応じて増減させる）
    parser.add_argument("-B", "--num_envs", type=int, default=num_envs)

    # 学習を回す最大のイテレーション（更新）回数。デフォルトは101回
    parser.add_argument("--max_iterations", type=int, default=num_iterations)

    args = parser.parse_args() # 入力された引数を解析してargsに格納

    # 2. ログフォルダと設定（Config）の準備
    log_dir = f"logs/{args.exp_name}" 
    env_cfg, obs_cfg, reward_cfg, command_cfg = PythonConfig.get_cfgs()
    train_cfg = PythonConfig.get_train_cfg(args.exp_name, args.max_iterations)

    # 3. 過去のログデータのクリーンアップ
    if os.path.exists(log_dir): # もし同じ名前のログフォルダがすでに存在していたら
        shutil.rmtree(log_dir)  # 過去のデータが混ざらないようにフォルダごと完全に削除する
    os.makedirs(log_dir, exist_ok=True) # 新しくログ保存用の空フォルダを作成する

    # 4. 学習設定のバックアップ保存
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], # 保存する設定データのリスト
        open(f"{log_dir}/cfgs.pkl", "wb"), # "wb"（バイナリ書き込みモード）でファイルを開いて保存する
    )

    gs.init(
        seed                = None,
        precision           = '32',
        debug               = False,
        eps                 = 1e-12,
        backend             = gs.gpu,
        theme               = 'dark',
        logging_level       = "warning",
        logger_verbose_time = False,
        performance_mode    = True
    )

    # 6. 並列環境のインスタンス化
    env = StandingEnv(
        num_envs=args.num_envs,     # 4096個の並列環境を作成
        env_cfg=env_cfg,            # 環境の基本設定を渡す
        obs_cfg=obs_cfg,            # 観測（状態）の設定を渡す
        reward_cfg=reward_cfg,      # 報酬関数の重みなどの設定を渡す
        command_cfg=command_cfg,     # 目標値（速度など）の設定を渡す
        show_viewer=False,  # 学習中はヘッドレスモードで動かす（ウィンドウを表示しない）
    )

    # 7. rsl_rlランナーの初期化と学習の開始
    # 作成した環境(env)、学習設定(train_cfg)、保存先(log_dir)、実行デバイス(GPU)を渡してPPOランナーを作成
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    # 実際の学習ループをスタート。指定した回数だけネットワークを更新する
    # init_at_random_ep_len=True は、各環境の初期化タイミングをずらし、全環境が同時にリセットされるのを防ぐ工夫
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


# --- スクリプトのエントリポイント ---
# このファイルが直接実行された時だけmain()関数を呼び出す（他のファイルからimportされた時は動かさない）
if __name__ == "__main__":
    main()