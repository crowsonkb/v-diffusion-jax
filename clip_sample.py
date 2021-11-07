#!/usr/bin/env python3

import argparse
from functools import partial
from pathlib import Path
import sys

from einops import repeat
import jax
import jax.numpy as jnp
from tqdm import tqdm, trange

from diffusion import get_model, get_models, load_params, sampling, utils

MODULE_DIR = Path(__file__).resolve().parent
sys.path.append(str(MODULE_DIR / 'CLIP_JAX'))

import clip_jax


def make_normalize(mean, std):
    mean = jnp.array(mean).reshape([3, 1, 1])
    std = jnp.array(std).reshape([3, 1, 1])

    def inner(image):
        return (image - mean) / std
    return inner


def norm2(x):
    """Normalizes a batch of vectors to the unit sphere."""
    return x / jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True))


def spherical_dist_loss(x, y):
    """Computes 1/2 the squared spherical distance between the two arguments."""
    return jnp.square(jnp.arccos(jnp.sum(norm2(x) * norm2(y), axis=-1))) / 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('prompt', type=str,
                   help='the text prompt')
    p.add_argument('--batch-size', '-bs', type=int, default=1,
                   help='the number of images per batch')
    p.add_argument('--checkpoint', type=str,
                   help='the checkpoint to use')
    p.add_argument('--clip-guidance-scale', '-cs', type=float, default=1000.,
                   help='the CLIP guidance scale')
    p.add_argument('--eta', type=float, default=1.,
                   help='the amount of noise to add during sampling (0-1)')
    p.add_argument('--model', type=str, choices=get_models(), required=True,
                   help='the model to use')
    p.add_argument('-n', type=int, default=1,
                   help='the number of images to sample')
    p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
    p.add_argument('--steps', type=int, default=1000,
                   help='the number of timesteps')
    args = p.parse_args()

    model = get_model(args.model)
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = MODULE_DIR / f'checkpoints/{args.model}.pkl'
    params = load_params(checkpoint)

    image_fn, text_fn, clip_params, _ = clip_jax.load('ViT-B/16')
    clip_patch_size = 16
    clip_size = 224
    normalize = make_normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])

    target_embed = text_fn(clip_params, clip_jax.tokenize([args.prompt]))

    key = jax.random.PRNGKey(args.seed)

    def clip_cond_fn_loss(x, key, params, clip_params, t, extra_args):
        dummy_key = jax.random.PRNGKey(0)
        v = model.apply(params, dummy_key, x, repeat(t, '-> n', n=x.shape[0]), extra_args)
        alpha, sigma = utils.t_to_alpha_sigma(t)
        pred = x * alpha - v * sigma
        clip_in = jax.image.resize(pred, (*pred.shape[:2], clip_size, clip_size), 'cubic')
        extent = clip_patch_size // 2
        clip_in = jnp.pad(clip_in, [(0, 0), (0, 0), (extent, extent), (extent, extent)], 'edge')
        sat_vmap = jax.vmap(partial(jax.image.scale_and_translate, method='cubic'),
                            in_axes=(0, None, None, 0, 0))
        scales = jnp.ones([pred.shape[0], 2])
        translates = jax.random.uniform(key, [pred.shape[0], 2], minval=-extent, maxval=extent)
        clip_in = sat_vmap(clip_in, (3, clip_size, clip_size), (1, 2), scales, translates)
        image_embeds = image_fn(clip_params, normalize((clip_in + 1) / 2))
        return jnp.sum(spherical_dist_loss(image_embeds, target_embed))

    def clip_cond_fn(x, key, t, extra_args, params, clip_params):
        grad_fn = jax.grad(clip_cond_fn_loss)
        grad = grad_fn(x, key, params, clip_params, t, extra_args)
        return grad * -args.clip_guidance_scale

    def run(key, n):
        tqdm.write('Sampling...')
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, [n, *model.shape])
        key, subkey = jax.random.split(key)
        cond_params = {'params': params, 'clip_params': clip_params}
        sample_step = partial(sampling.jit_cond_sample_step,
                              extra_args={},
                              cond_fn=clip_cond_fn,
                              cond_params=cond_params)
        steps = utils.get_ddpm_schedule(jnp.linspace(1, 0, args.steps + 1)[:-1])
        return sampling.sample_loop(model, params, subkey, noise, steps, args.eta, sample_step)

    def run_all(key, n, batch_size):
        for i in trange(0, n, batch_size):
            key, subkey = jax.random.split(key)
            cur_batch_size = min(n - i, batch_size)
            outs = run(key, cur_batch_size)
            for j, out in enumerate(outs):
                utils.to_pil_image(out).save(f'out_{i + j:05}.png')

    try:
        run_all(key, args.n, args.batch_size)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
