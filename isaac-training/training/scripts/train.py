# training/scripts/train.py
import sys, os, argparse

# Ensure project root (the folder that contains "training/") is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    p = argparse.ArgumentParser()
    # core run controls
    p.add_argument("--headless", type=str, default="True")  # "True"/"False"
    p.add_argument("--sim_device", type=str, default="cuda:0")
    p.add_argument("--pipeline", type=str, default="gpu")
    p.add_argument("--num_envs", type=int, default=2048)
    p.add_argument("--max_iterations", type=int, default=3000)
    # wandb
    p.add_argument("--wandb.activate", dest="wandb_activate", type=str, default="True")
    p.add_argument("--wandb.project",  dest="wandb_project",  type=str, default="isaac-cartpole")
    p.add_argument("--wandb.entity",   dest="wandb_entity",   type=str, default=None)
    p.add_argument("--wandb.group",    dest="wandb_group",    type=str, default="RTX4070")
    p.add_argument("--wandb.name",     dest="wandb_name",     type=str, default="cartpole_run")

    args, unknown = p.parse_known_args()

    # Try to get TASK_MAP from OIGE (different tags put it in different places)
    TASK_MAP = None
    try:
        from omniisaacgymenvs.tasks.task_map import TASK_MAP as _TASK_MAP
        TASK_MAP = _TASK_MAP
    except Exception:
        try:
            # some versions used utils
            from omniisaacgymenvs.utils.task_util import TASK_MAP as _TASK_MAP
            TASK_MAP = _TASK_MAP
        except Exception:
            TASK_MAP = None

    # Try to override Cartpole with our local task (optional; continue if it fails)
    if TASK_MAP is not None:
        try:
            from training.envs.cartpole_task import CartpoleTask
            TASK_MAP["Cartpole"] = CartpoleTask
            print("[info] Overrode OIGE Cartpole with training.envs.cartpole_task.CartpoleTask")
        except Exception as e:
            print(f"[warn] Could not override Cartpole task ({e}). Using upstream OIGE Cartpole.")
    else:
        print("[warn] Could not import TASK_MAP from omniisaacgymenvs; using upstream OIGE Cartpole.")

    # Build Hydra-style argv for OIGE trainer
    hydra = [
        "task=Cartpole",
        f"headless={args.headless}",
        f"sim_device={args.sim_device}",
        f"pipeline={args.pipeline}",
        f"num_envs={args.num_envs}",
        f"max_iterations={args.max_iterations}",
        f"wandb_activate={args.wandb_activate}",
        f"wandb_project={args.wandb_project}",
        f"wandb_group={args.wandb_group}",
        f"wandb_name={args.wandb_name}",
    ]
    if args.wandb_entity:
        hydra.append(f"wandb_entity={args.wandb_entity}")
    hydra += unknown

    # Hand off to OIGE’s rl-games trainer
    from omniisaacgymenvs.scripts.rlgames_train import main as rlgames_main
    sys.argv = [sys.argv[0]] + hydra
    rlgames_main()

if __name__ == "__main__":
    main()
