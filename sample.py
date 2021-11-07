#!/usr/bin/env python3

"""Unconditional sampling from a diffusion model."""

import argparse
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from tqdm import tqdm, trange

from diffusion import get_model, get_models, load_params, sampling, utils

MODULE_DIR = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', '-bs', type=int, default=1,
                   help='the number of images per batch')
    p.add_argument('--checkpoint', type=str,
                   help='the checkpoint to use')
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

    key = jax.random.PRNGKey(args.seed)

    def run(key, n):
        tqdm.write('Sampling...')
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, [n, *model.shape])
        key, subkey = jax.random.split(key)
        sample_step = partial(sampling.jit_sample_step, extra_args={})
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
