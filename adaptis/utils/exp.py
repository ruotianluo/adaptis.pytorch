import os
import shutil
import logging
from datetime import datetime
from pathlib import Path

import torch

from adaptis.utils.log import logger
from adaptis.utils.args import get_train_arguments


def init_experiment(experiment_name, add_exp_args, script_path=None):
    parser = get_train_arguments()
    parser = add_exp_args(parser)
    args = parser.parse_args()

    experiments_path = Path('./experiments') / experiment_name
    experiments_path.mkdir(parents=True, exist_ok=True)

    if args.resume:
        experiment_name = args.resume_exp_name
    else:
        exp_indx = find_last_exp_indx(experiments_path)
        experiment_name = f'{exp_indx:03d}'
        if args.exp_name:
            experiment_name += f'_{args.exp_name}'

    experiment_path = experiments_path / experiment_name

    args.logs_path = experiment_path / 'logs'
    args.run_path = experiment_path
    args.checkpoints_path = experiment_path / 'checkpoints'

    if not os.path.exists(experiment_path):
        experiment_path.mkdir(parents=True)
    if script_path is not None:
        temp_script_name = Path(script_path).stem + datetime.strftime(datetime.today(), '_%Y-%m-%d_%H-%M-%S.py')
        shutil.copy(script_path, experiment_path / temp_script_name)

    if not args.checkpoints_path.exists():
        args.checkpoints_path.mkdir(parents=True)
    if not args.logs_path.exists():
        args.logs_path.mkdir(parents=True)

    stdout_log_path = args.logs_path / 'train_log.txt'

    if stdout_log_path is not None:
        fh = logging.FileHandler(str(stdout_log_path))
        formatter = logging.Formatter(fmt='(%(levelname)s) %(asctime)s: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if args.no_cuda:
        logger.info('Using CPU')
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda')
        if args.gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpus}"
            args.ngpus = len(args.gpus.split(','))
            logger.info(f'Number of GPUs: {args.ngpus}')
        else:
            args.ngpus = torch.cuda.device_count()

        if args.ngpus > 1:
            assert args.dist, '--dist need to be on for multi-gpu training'

        if args.ngpus < 2:
            args.syncbn = False

    logger.info(args)

    return args


def find_last_exp_indx(exp_parent_path):
    indx = 0
    for x in exp_parent_path.iterdir():
        if not x.is_dir():
            continue

        exp_name = x.stem
        if exp_name[:3].isnumeric():
            indx = max(indx, int(exp_name[:3]) + 1)

    return indx
