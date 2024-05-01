import argparse

import yaml

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to the sweep config file")
parser.add_argument("--sweep_id", help="Sweep ID")
args = parser.parse_args()

if args.config:
    sweep_config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    sweep_id = wandb.sweep(sweep_config)
elif args.sweep_id:
    sweep_id = args.sweep_id
else:
    raise ValueError("Either --config or --sweep_id must be provided")

wandb.agent(sweep_id)
