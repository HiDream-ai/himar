import math
import sys
from typing import Iterable
import torch.nn.functional as F
import torch

import util.misc as misc
import util.lr_sched as lr_sched
from models.vae import DiagonalGaussianDistribution
import torch_fidelity
import shutil
import cv2
import numpy as np
import os
import copy
import time
from scipy.ndimage import zoom
from fid_score import calculate_fid_given_paths
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/"+'{}.png'.format(str(i).zfill(5)))
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def train_one_epoch(model, vae,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,vq_model=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, cond, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            if args.use_cached:
                moments = samples
                posterior = DiagonalGaussianDistribution(moments)
            else:
                posterior = vae.encode(samples)

            # normalize the std of latent to be 1. Change it if you use a different tokenizer
            x = posterior.sample().mul_(0.2325)
            if args.use_cached:
                moments = cond
                posterior = DiagonalGaussianDistribution(moments)
            else:
                posterior = vae.encode(cond)

            cond = posterior.sample().mul_(0.2325)

        # forward
        with torch.cuda.amp.autocast(): #type=torch.bfloat16):
            loss1, loss2 = model(x, labels,cond=cond)
        loss = (64.0/320) * loss1 + (256.0/320) * loss2 
        
        loss_value = loss.item()
        loss_value1 = loss1.item()
        loss_value2 = loss2.item()
      

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        update_ema(ema_params, model_params, rate=args.ema_rate)

        metric_logger.update(loss=loss_value,loss_s=loss_value1,loss_l=loss_value2)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_value_reduce1 = misc.all_reduce_mean(loss_value1)
        loss_value_reduce2 = misc.all_reduce_mean(loss_value2)
        
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_s', loss_value_reduce1, epoch_1000x)
            log_writer.add_scalar('train_loss_l', loss_value_reduce2, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=16, log_writer=None, cfg=1.0,
             use_ema=True, vq_model=None, re_cfg=1.0):
    model_without_ddp.eval()
    num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1
    save_folder = os.path.join(args.output_dir, "ariter{}-diffsteps{}-temp{}-{}cfg{}-image{}".format(args.num_iter,
                                                                                                     args.num_sampling_steps,
                                                                                                     args.temperature,
                                                                                                     args.cfg_schedule,
                                                                                                     cfg,
                                                                                                     args.num_images))
    if use_ema:
        save_folder = save_folder + "_ema"
    if args.evaluate:
        save_folder = save_folder + "_evaluate"
    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder+'small'):
            os.makedirs(save_folder+'small')
        if not os.path.exists(save_folder+'big_cond'):
            os.makedirs(save_folder+'big_cond')

            

    # switch to ema params
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    class_num = args.class_num
    assert args.num_images % class_num == 0  # number of images per class must be the same
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    used_time = 0
    gen_img_cnt = 0
    device = torch.device("cuda")
    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        labels_gen = class_label_gen_world[world_size * batch_size * i + local_rank * batch_size:
                                                world_size * batch_size * i + (local_rank + 1) * batch_size]
        labels_gen = torch.Tensor(labels_gen).long().cuda()


        torch.cuda.synchronize()
        start_time = time.time()

        # generation
        with torch.no_grad():
            with torch.cuda.amp.autocast(): #dtype=torch.bfloat16):
                small_sampled_tokens, big_cond_sampled_tokens = model_without_ddp.sample_tokens(bsz=batch_size, num_iter=args.num_iter, cfg=cfg,re_cfg=args.re_cfg,
                                                                cfg_schedule=args.cfg_schedule, labels=labels_gen,
                                                                temperature=args.temperature,vq_model=vq_model,
                                                                )
                small_sampled_images = vae.decode(small_sampled_tokens/ 0.2325)
                
                small_sampled_images = F.interpolate(small_sampled_images, size=(64, 64), mode='bicubic')
                big_cond_sampled_images = vae.decode(big_cond_sampled_tokens / 0.2325)
                
        # measure speed after the first generation batch
        if i >= 1:
            torch.cuda.synchronize()
            used_time += time.time() - start_time
            gen_img_cnt += batch_size
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(gen_img_cnt, used_time, used_time / gen_img_cnt))

        torch.distributed.barrier()
        small_sampled_images = small_sampled_images.detach().cpu()
        small_sampled_images = (small_sampled_images + 1) / 2
        big_cond_sampled_images = big_cond_sampled_images.detach().cpu()
        big_cond_sampled_images = (big_cond_sampled_images + 1) / 2
        sampled_images = small_sampled_images


        # distributed save
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(small_sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder+'small', '{}.png'.format(str(img_id).zfill(5))), gen_img)

            gen_img = np.round(np.clip(big_cond_sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder+'big_cond', '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()
    time.sleep(10)

    # back to no ema
    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)
    if log_writer is not None:
        if args.img_size == 256:
            input2 = None
            fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
        else:
            raise NotImplementedError

        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder+'big_cond',
            input2=input2,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        print("cond_fid:",fid)
        print("cond_inception_score",inception_score)
        fid_statistics_file = 'fid_stats/fid_stats_imagenet64_guided_diffusion.npz'
        small_fid = calculate_fid_given_paths((fid_statistics_file, save_folder+'small'))
        print("small_fid: ",small_fid)

        
        postfix = ""
        if use_ema:
           postfix = postfix + "_ema"
        if not cfg == 1.0:
           postfix = postfix + "_cfg{}".format(cfg)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        # remove temporal saving folder
        shutil.rmtree(save_folder+'small')
        shutil.rmtree(save_folder+'big_cond')

    torch.distributed.barrier()
    time.sleep(10)
    

def cache_latents_w_noise(vae,
                  data_loader: Iterable,
                  device: torch.device,
                  args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 20

    for data_iter_step, (samples, _, paths) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)

        with torch.no_grad():
            posterior = vae.encode(samples)
            moments = posterior.parameters
            posterior_flip = vae.encode(samples.flip(dims=[3]))
            moments_flip = posterior_flip.parameters
            x = posterior.sample()
            small_samples = vae.decode(x)
            small_samples = F.interpolate(small_samples, size=(64, 64), mode='bicubic')
            small_samples = small_samples.clamp(-1,1)
            posterior = vae.encode(small_samples)
            small_moments = posterior.parameters
            flip_x = posterior_flip.sample()
            flip_small_samples = vae.decode(flip_x)
            flip_small_samples = F.interpolate(flip_small_samples, size=(64, 64), mode='bicubic')
            flip_small_samples = flip_small_samples.clamp(-1,1)
            posterior_flip = vae.encode(flip_small_samples)
            small_moments_flip = posterior_flip.parameters

        for i, path in enumerate(paths):
            save_path = os.path.join(args.cached_path, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, moments=moments[i].cpu().numpy(), 
                     moments_flip=moments_flip[i].cpu().numpy(),
                     small_moments=small_moments[i].cpu().numpy(), 
                     small_moments_flip=small_moments_flip[i].cpu().numpy(),
                     )

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

    return

def cache_latents(vae,
                  data_loader: Iterable,
                  device: torch.device,
                  small_scale: int,
                  args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 20
    device = torch.device("cuda")
    for data_iter_step, (samples, _, paths) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)

        with torch.no_grad():
            posterior = vae.encode(samples)
            moments = posterior.parameters
            posterior_flip = vae.encode(samples.flip(dims=[3]))
            moments_flip = posterior_flip.parameters
            small_samples = F.interpolate(samples, size=(small_scale, small_scale), mode='bicubic')
            posterior = vae.encode(small_samples)
            small_moments = posterior.parameters
            posterior_flip = vae.encode(small_samples.flip(dims=[3]))
            small_moments_flip = posterior_flip.parameters


        for i, path in enumerate(paths):
            save_path = os.path.join(args.cached_path, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, moments=moments[i].cpu().numpy(), 
                     moments_flip=moments_flip[i].cpu().numpy(),
                     small_moments=small_moments[i].cpu().numpy(), 
                     small_moments_flip=small_moments_flip[i].cpu().numpy(),
                     )

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

    return
