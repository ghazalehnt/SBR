import argparse
import json
import os
import random
from os.path import exists, join

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import numpy as np
import torch
from tensorboardX import SummaryWriter

from data_loading import load_data
from others import get_model
from supervised import SupervisedTrainer


def training_function(tuning_config, stationary_config_file, exp_root_dir, data_root_dir,
                      valid_metric, early_stopping_patience=None):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    # TODO
    metrics = ["ndcg_cut_5", "P_5", 'precision_micro', 'recall_micro', 'f1_micro', 'precision_weighted',
               'recall_weighted', 'f1_weighted', 'precision_macro', 'recall_macro', 'f1_macro']

    config = json.load(open(stationary_config_file, 'r'))
    for k in tuning_config:
        if '.' in k:
            l1 = k[:k.index(".")]
            l2 = k[k.index(".")+1:]
            config[l1][l2] = tuning_config[k]
        else:
            config[k] = tuning_config[k]
    if "<DATA_ROOT_PATH>" in config["dataset"]["dataset_path"]:
        config["dataset"]["dataset_path"] = config["dataset"]["dataset_path"] \
            .replace("<DATA_ROOT_PATH>", data_root_dir)
    if "<EXP_ROOT_PATH>" in config["experiment_root"]:
        config["experiment_root"] = config["experiment_root"] \
            .replace("<EXP_ROOT_PATH>", exp_root_dir)

    exp_dir_params = []
    for param in config['params_in_exp_dir']:
        p1 = param[:param.index(".")]
        p2 = param[param.index(".") + 1:]
        exp_dir_params.append(str(config[p1][p2]))
    exp_dir = join(config['experiment_root'], "_".join(exp_dir_params))
    config["experiment_dir"] = exp_dir
    if early_stopping_patience is not None:
        config['trainer']['early_stopping_patience'] = early_stopping_patience
    config['trainer']['valid_metric'] = valid_metric
    print(config)

    if exists(exp_dir):
        saved_config = json.load(open(os.path.join(exp_dir, "config.json"), 'r'))
        if saved_config != config:
            raise ValueError(f"Given config should be the same as saved config file!!!!{exp_dir}")
    else:
        os.makedirs(exp_dir)
        with open(os.path.join(exp_dir, "config.json"), 'w') as log:
            json.dump(config, log, indent=4)

    # log into the ray_tune trial folder
    with open("config.json", 'w') as log:
        json.dump(config, log, indent=4)

    # todo do we need all of these??? dts
    logger = SummaryWriter(exp_dir)
    for k, v in config["dataset"].items():
        logger.add_text(f"dataset/{k}", str(v))
    for k, v in config["trainer"].items():
        logger.add_text(f"trainer/{k}", str(v))
    for k, v in config["model"].items():
        logger.add_text(f"model/{k}", str(v))
    logger.add_text("exp_dir", exp_dir)
    print("experiment_dir:")
    print(exp_dir)

    train_dataloader, valid_dataloader, test_dataloader, users, items, relevance_level, padding_token = \
        load_data(config['dataset'],
                  config['model']['pretrained_model'] if 'pretrained_model' in config['model'] else None)

    prec_path = None
    if 'pretrained_model' in config['model']:
        prec_path = join(config['dataset']['dataset_path'], 'precomputed_reps',
                         f"size{config['dataset']['chunk_size']}_u{config['dataset']['max_num_chunks_user']}-"
                         f"{'-'.join(config['dataset']['user_text'])}_{config['dataset']['user_review_choice']}_"
                         f"i{config['dataset']['max_num_chunks_item']}-{'-'.join(config['dataset']['item_text'])}")
        os.makedirs(prec_path, exist_ok=True)
    model = get_model(config['model'], users, items,
                      1 if config['dataset']['binary_interactions'] else None, padding_token, device,
                      prec_path)  # todo else num-ratings

    trainer = SupervisedTrainer(config=config['trainer'], model=model, device=device, logger=logger,
                                exp_dir=exp_dir, test_only=False, tuning=True,
                                relevance_level=relevance_level, users=users, items=items)
    trainer.fit(train_dataloader, valid_dataloader)


def main(hyperparameter_config, config_file, ray_result_dir, name, valid_metric, max_epochs=50, grace_period=5, num_gpus_per_trial=0,
         num_cpus_per_trial=2, extra_gpus=0, num_samples=1, resume=False, early_stopping_patience=None):
    exp_root_dir = open("data/paths_vars/EXP_ROOT_PATH").read().strip()
    data_root_dir = open("data/paths_vars/DATA_ROOT_PATH").read().strip()
    if "<EXP_ROOT_PATH>" in ray_result_dir:
        ray_result_dir = ray_result_dir.replace("<EXP_ROOT_PATH>", exp_root_dir)
    print(f"ray dir: {ray_result_dir}")
    scheduler = ASHAScheduler(
        metric="best_valid_metric",
        mode="min" if "loss" in valid_metric else "max",
        max_t=max_epochs,
        grace_period=grace_period
    )
    reporter = CLIReporter(
        metric_columns=["epoch", "best_valid_metric", "best_epoch"], max_report_frequency=3600
    )
    result = tune.run(
        tune.with_parameters(training_function, stationary_config_file=config_file,
                             valid_metric=valid_metric, early_stopping_patience=early_stopping_patience,
                             exp_root_dir=exp_root_dir, data_root_dir=data_root_dir),
        name=name,
        resources_per_trial={"cpu": num_cpus_per_trial, "gpu": num_gpus_per_trial, "extra_gpu": extra_gpus},
        config=hyperparameter_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=ray_result_dir,
        resume=resume,
        # max_concurrent_trials=1  # todo remove - set 1 for debug
    )
    best_trial = result.get_best_trial("best_valid_metric", "min" if "loss" in valid_metric else "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final best_valid_metric({valid_metric}): {best_trial.last_result['best_valid_metric']} - "
          f"best_epoch: {best_trial.last_result['best_epoch']}, last_epoch: {best_trial.last_result['epoch']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the parameter tuning pipeline.')
    parser.add_argument('--config_file', type=str, help='path to configuration file.')
    parser.add_argument('--hyperparam_config_file', type=str, help='path to hyperparam configuration file.')
    parser.add_argument('--num_gpu', type=int, help='number of gpus.')

    args = parser.parse_args()
    if not exists(args.config_file):
        raise ValueError(f"File: {args.config_file} does not exist!")
    if not exists(args.hyperparam_config_file):
        raise ValueError(f"File: {args.hyperparam_config_file} does not exist!")

    config = json.load(open(args.hyperparam_config_file, 'r'))

    hyper_config = {}
    for k, v in config["space"].items():
        if 'grid_search' in v:
            hyper_config[k] = tune.grid_search(v['grid_search'])
        elif 'quniform' in v:
            hyper_config[k] = tune.quniform(v['quniform'][0], v['quniform'][1], v['quniform'][2])
        elif 'choice' in v:
            hyper_config[k] = tune.choice(v['choice'])
        else:
            raise NotImplemented("implement different space types")

    ray.init(num_gpus=args.num_gpu)
    main(hyper_config, os.path.abspath(args.config_file), config["ray_result_dir"],
         config["name"], config["valid_metric"],
         max_epochs=config["max_epochs"], grace_period=config["grace_period"],
         num_gpus_per_trial=config["num_gpus_per_trial"], num_cpus_per_trial=config["num_cpus_per_trial"],
         num_samples=config["num_samples"], resume=config["resume"],
         early_stopping_patience=config["early_stopping_patience"] if "early_stopping_patience" in config else None)
