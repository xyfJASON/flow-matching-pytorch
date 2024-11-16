import argparse
import math
import os
from omegaconf import OmegaConf

import accelerate
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import EMA, FlowMatchingSampler
from utils import (
    load_data, get_logger, create_exp_dir, find_resume_checkpoint, instantiate_from_config,
    get_time_str, check_freq, get_dataloader_iterator, discard_label, StatusTracker,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('-e', '--exp_dir', type=str, help='Path to the experiment directory. Default to be ./runs/exp-{current time}/')
    parser.add_argument('-r', '--resume', type=str, help='Resume from a checkpoint. Could be a path or `best` or `latest`')
    parser.add_argument('-mp', '--mixed_precision', type=str, default=None, help='Mixed precision training')
    parser.add_argument('-cd', '--cover_dir', action='store_true', default=False, help='Cover the experiment directory if it exists')
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator(
        step_scheduler_with_optimizer=False,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()

    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if accelerator.is_main_process:
        create_exp_dir(
            exp_dir=exp_dir, conf_yaml=OmegaConf.to_yaml(conf), subdirs=['ckpt', 'samples'],
            time_str=args.time_str, exist_ok=args.resume is not None, cover_dir=args.cover_dir,
        )

    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True, is_main_process=accelerator.is_main_process,
    )

    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger, print_freq=conf.train.print_freq,
        tensorboard_dir=os.path.join(exp_dir, 'tensorboard'),
        is_main_process=accelerator.is_main_process,
    )

    # SET SEED
    accelerate.utils.set_seed(conf.seed, device_specific=True)
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')
    accelerator.wait_for_everyone()

    # BUILD DATASET AND DATALOADER
    assert conf.train.batch_size % accelerator.num_processes == 0
    bspp = conf.train.batch_size // accelerator.num_processes
    train_set = load_data(conf.data, split='all' if conf.data.name.lower() == 'ffhq' else 'train')
    train_loader = DataLoader(train_set, batch_size=bspp, shuffle=True, drop_last=True, **conf.dataloader)
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Batch size per process: {bspp}')
    logger.info(f'Total batch size: {conf.train.batch_size}')

    # BUILD MODEL
    model = instantiate_from_config(conf.model)
    ema = EMA(model.parameters(), **conf.train.ema)
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Number of parameters of model: {sum(p.numel() for p in model.parameters()):,}')
    logger.info('=' * 50)

    # BUILD OPTIMIZER AND SCHEDULER
    optimizer = instantiate_from_config(conf.train.optim, params=model.parameters())
    scheduler = instantiate_from_config(conf.train.sched, optimizer=optimizer)

    # BUILD FLOW MATCHING SAMPLER
    sampler = FlowMatchingSampler()

    # RESUME TRAINING
    step = 0
    if args.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, args.resume)
        logger.info(f'Resume from {resume_path}')
        # load model
        ckpt_model = torch.load(os.path.join(resume_path, 'model.pt'), map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt_model['model'])
        logger.info(f'Successfully load model from {resume_path}')
        # load ema
        ckpt_ema = torch.load(os.path.join(resume_path, 'ema.pt'), map_location='cpu', weights_only=True)
        ema.load_state_dict(ckpt_ema['ema'])
        logger.info(f'Successfully load ema from {resume_path}')
        # load optimizer
        ckpt_optimizer = torch.load(os.path.join(resume_path, 'optimizer.pt'), map_location='cpu', weights_only=True)
        optimizer.load_state_dict(ckpt_optimizer['optimizer'])
        logger.info(f'Successfully load optimizer from {resume_path}')
        # load scheduler
        ckpt_scheduler = torch.load(os.path.join(resume_path, 'scheduler.pt'), map_location='cpu', weights_only=True)
        scheduler.load_state_dict(ckpt_scheduler['scheduler'])
        logger.info(f'Successfully load scheduler from {resume_path}')
        # load meta information
        ckpt_meta = torch.load(os.path.join(resume_path, 'meta.pt'), map_location='cpu', weights_only=True)
        step = ckpt_meta['step'] + 1
        logger.info(f'Restart training at step {step}')
        del ckpt_model, ckpt_optimizer, ckpt_scheduler, ckpt_meta

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    model, optimizer, scheduler, train_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader,  # type: ignore
    )
    unwrapped_model = accelerator.unwrap_model(model)
    ema.to(device)
    accelerator.wait_for_everyone()

    # TRAINING FUNCTIONS
    @accelerator.on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save model and ema model
        accelerator.save(dict(model=unwrapped_model.state_dict()), os.path.join(save_path, 'model.pt'))
        with ema.scope(model.parameters()):
            accelerator.save(dict(model=unwrapped_model.state_dict()), os.path.join(save_path, 'model_ema.pt'))
        # save ema
        accelerator.save(dict(ema=ema.state_dict()), os.path.join(save_path, 'ema.pt'))
        # save optimizer
        accelerator.save(dict(optimizer=optimizer.state_dict()), os.path.join(save_path, 'optimizer.pt'))
        # save scheduler
        accelerator.save(dict(scheduler=scheduler.state_dict()), os.path.join(save_path, 'scheduler.pt'))
        # save meta information
        accelerator.save(dict(step=step), os.path.join(save_path, 'meta.pt'))

    def run_step(batch):
        # get data
        x = discard_label(batch).float()
        B = x.shape[0]

        with accelerator.autocast():
            # straight flow
            timesteps = torch.rand((B, ), device=device, dtype=torch.float)
            noise = torch.randn_like(x)
            x_inter = (1 - timesteps.reshape(-1, 1, 1, 1)) * x + timesteps.reshape(-1, 1, 1, 1) * noise
            # model forward
            model_output = model(x_inter, timesteps * 999)
            # mse loss (predict the velocity)
            loss = F.mse_loss(model_output, noise - x)

        # backward
        accelerator.backward(loss)

        # optimize
        optimizer.step()
        scheduler.step()
        ema.update(model.parameters())
        optimizer.zero_grad()
        return dict(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    @torch.no_grad()
    def sample(savepath):
        nrow = math.ceil(math.sqrt(conf.train.n_samples))
        n_samples = math.ceil(conf.train.n_samples / accelerator.num_processes)
        init_noise = torch.randn(n_samples, 3, conf.data.img_size, conf.data.img_size, device=device)
        with ema.scope(model.parameters()):
            samples = sampler.sample(model, init_noise, sampling_steps=100).clamp(-1, 1)
        samples = accelerator.gather(samples)[:conf.train.n_samples].cpu()
        if accelerator.is_main_process:
            save_image(samples, savepath, nrow=nrow, normalize=True, value_range=(-1, 1))

    # START TRAINING
    logger.info('Start training...')
    train_loader_iterator = get_dataloader_iterator(
        dataloader=train_loader,
        tqdm_kwargs=dict(desc='Epoch', leave=False, disable=not accelerator.is_main_process),
    )
    while step < conf.train.n_steps:
        # get a batch of data
        _batch = next(train_loader_iterator)
        # run a step
        model.train()
        train_status = run_step(_batch)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()
        # validate
        model.eval()
        # save checkpoint
        if check_freq(conf.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>7d}'))
            accelerator.wait_for_everyone()
        # sample from current model
        if check_freq(conf.train.sample_freq, step):
            sample(os.path.join(exp_dir, 'samples', f'step{step:0>7d}.png'))
            accelerator.wait_for_everyone()
        step += 1
    # save the last checkpoint if not saved
    if not check_freq(conf.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>7d}'))
    accelerator.wait_for_everyone()
    status_tracker.close()
    accelerator.end_training()
    logger.info('End of training')


if __name__ == '__main__':
    main()
