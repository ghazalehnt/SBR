import argparse
import datetime
import json
import os
from os.path import join

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from SBR.model.mf_dot import MatrixFactorizatoinDotProduct
from SBR.trainer.supervised import SupervisedTrainer
from SBR.utils.data_loading import load_data


def main(op, config_file=None, result_folder=None):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_only = False
    if op == "train":
        config = json.load(open(config_file, 'r'))
        exp_dir_params = []
        for param in config['params_in_exp_dir']:
            p1 = param[:param.index(".")]
            p2 = param[param.index(".")+1:]
            exp_dir_params.append(str(config[p1][p2]))
        exp_dir = join(config['experiment_root'], "_".join(exp_dir_params))
        config["experiment_dir"] = exp_dir
        # check if the exp dir exists, the config file is the same as given.
        if os.path.exists(join(exp_dir, "config.json")):
            config2 = json.load(open(join(exp_dir, "config.json"), 'r'))
            if config != config2:
                raise ValueError(f"{exp_dir} exists with different config != {config_file}")
        os.makedirs(exp_dir, exist_ok=True)
        json.dump(config, open(join(exp_dir, "config.json"), 'w'))
    elif op == "test":
        config = json.load(open(join(result_folder, "config.json"), 'r'))
        test_only = True
        exp_dir = config["experiment_dir"]
    else:
        raise ValueError("op not defined!")

    # todo do we need all of these??? dts
    logger = SummaryWriter(exp_dir)
    for k, v in config["dataset"].items():
        logger.add_text(f"dataset/{k}", str(v))
    for k, v in config["trainer"].items():
        logger.add_text(f"trainer/{k}", str(v))
    for k, v in config["model"].items():
        logger.add_text(f"model/{k}", str(v))
    logger.add_text("exp_dir", exp_dir)

    train_dataloader, valid_dataloader, test_dataloader, users, items, relevance_level = load_data(config['dataset'])

    # TODO
    model = MatrixFactorizatoinDotProduct(config=config['model'], n_users=users.num_rows, n_items=items.num_rows)

    trainer = SupervisedTrainer(config=config['trainer'], model=model, device=device, logger=logger, exp_dir=exp_dir,
                                test_only=test_only, relevance_level=relevance_level)

    if op == "train":
        trainer.fit(train_dataloader, valid_dataloader)
        trainer.evaluate(test_dataloader, valid_dataloader)
    elif op == "test":
        trainer.evaluate(test_dataloader, valid_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-c', type=str, default=None, help='config file, to train')
    parser.add_argument('--result_folder', '-r', type=str, default=None, help='result forler, to evaluate')
    parser.add_argument('--op', type=str, default=None, help='operation train/test')
    args, _ = parser.parse_known_args()

    if args.op == "train":
        if not os.path.exists(args.config_file):
            raise ValueError(f"Config file does not exist: {args.config_file}")
        if args.result_folder:
            raise ValueError(f"OP==train does not accept result_folder")
        main(op=args.op, config_file=args.config_file)
    elif args.op == "test":
        if not os.path.exists(join(args.result_folder, "config.json")):
            raise ValueError(f"Result folder does not exist: {args.config_file}")
        if args.config_file:
            raise ValueError(f"OP==test does not accept config_file")
        main(op=args.op, result_folder=args.result_folder)



