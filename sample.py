import argparse
import math
import os
import tqdm
from omegaconf import OmegaConf

import accelerate
import torch

from models import FlowMatchingSampler
from utils import get_logger, image_norm_to_float, save_images, instantiate_from_config, amortize


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to inference configuration file')
    parser.add_argument('--weights', type=str, required=True, help='Path to pretrained transformer weights')
    parser.add_argument('--n_samples', type=int, required=True, help='Number of samples')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to directory saving samples')
    parser.add_argument('--sampling_steps', type=int, default=100, help='Number of sampling steps')
    parser.add_argument('--sampling_method', type=str, default='euler', help='Sampling method')
    parser.add_argument('--bspp', type=int, default=100, help='Batch size on each process')
    parser.add_argument('--seed', type=int, default=8888, help='Set random seed')
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()

    # INITIALIZE LOGGER
    logger = get_logger(use_tqdm_handler=True, is_main_process=accelerator.is_main_process)

    # SET SEED
    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')
    accelerator.wait_for_everyone()

    # BUILD MODEL AND LOAD WEIGHTS
    model = instantiate_from_config(conf.model).eval().to(device)
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load model from {args.weights}')
    logger.info(f'Number of parameters of model: {sum(p.numel() for p in model.parameters()):,}')
    logger.info('=' * 50)

    # BUILD FLOW MATCHING SAMPLER
    sampler = FlowMatchingSampler(method=args.sampling_method)
    accelerator.wait_for_everyone()

    # SAMPLING FUNCTIONS
    logger.info('Start sampling...')
    logger.info(f'Samples will be saved to {args.save_dir}')
    os.makedirs(args.save_dir, exist_ok=True)
    cnt = 0
    bslist = amortize(args.n_samples, args.bspp * accelerator.num_processes)
    for bs in tqdm.tqdm(bslist, desc='Sampling', disable=not accelerator.is_main_process):
        bspp = min(args.bspp, math.ceil(bs / accelerator.num_processes))
        init_noise = torch.randn(bspp, 3, conf.data.img_size, conf.data.img_size, device=device)
        with torch.no_grad():
            samples = sampler.sample(model, init_noise, sampling_steps=args.sampling_steps).clamp(-1, 1)
        samples = accelerator.gather(samples)[:bs]
        if accelerator.is_main_process:
            samples = image_norm_to_float(samples).cpu()
            save_images(samples, args.save_dir, start_idx=cnt)
            cnt += bs
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')
    accelerator.end_training()


if __name__ == '__main__':
    main()
