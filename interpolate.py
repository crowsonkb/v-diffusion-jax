#!/usr/bin/env python3

"""Interpolation in a diffusion model's latent space."""

import argparse
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from PIL import Image
from tqdm import trange

from diffusion import get_model, get_models, load_params, sampling, utils

MODULE_DIR = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', '-bs', type=int, default=4,
                   help='the number of images per batch')
    p.add_argument('--checkpoint', type=str,
                   help='the checkpoint to use')
    p.add_argument('--init-1', type=str,
                   help='the init image for the starting point')
    p.add_argument('--init-2', type=str,
                   help='the init image for the ending point')
    p.add_argument('--model', type=str, choices=get_models(), required=True,
                   help='the model to use')
    p.add_argument('-n', type=int, default=16,
                   help='the number of images to sample')
    p.add_argument('--seed-1', type=int, default=0,
                   help='the random seed for the starting point')
    p.add_argument('--seed-2', type=int, default=1,
                   help='the random seed for the ending point')
    p.add_argument('--steps', type=int, default=1000,
                   help='the number of timesteps')
    args = p.parse_args()

    model = get_model(args.model)
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = MODULE_DIR / f'checkpoints/{args.model}.pkl'
    params = load_params(checkpoint)

    key_1 = jax.random.PRNGKey(args.seed_1)
    key_2 = jax.random.PRNGKey(args.seed_2)
    latent_1 = jax.random.normal(key_1, [1, *model.shape])
    latent_2 = jax.random.normal(key_2, [1, *model.shape])

    _, y, x = model.shape

    reverse_sample_step = partial(sampling.jit_reverse_sample_step, extra_args={})
    reverse_steps = utils.get_ddpm_schedule(jnp.linspace(0, 1, args.steps + 1))

    if args.init_1:
        init_1 = Image.open(args.init_1).convert('RGB').resize((x, y), Image.LANCZOS)
        init_1 = utils.from_pil_image(init_1)[None]
        print('Inverting the starting init image...')
        latent_1 = sampling.reverse_sample_loop(model, params, key_1, init_1, reverse_steps,
                                                reverse_sample_step)

    if args.init_2:
        init_2 = Image.open(args.init_2).convert('RGB').resize((x, y), Image.LANCZOS)
        init_2 = utils.from_pil_image(init_2)[None]
        print('Inverting the ending init image...')
        latent_2 = sampling.reverse_sample_loop(model, params, key_2, init_2, reverse_steps,
                                                reverse_sample_step)

    def run(weights):
        alphas, sigmas = utils.t_to_alpha_sigma(weights)
        latents = latent_1 * alphas[:, None, None, None] + latent_2 * sigmas[:, None, None, None]
        sample_step = partial(sampling.jit_sample_step, extra_args={})
        steps = utils.get_ddpm_schedule(jnp.linspace(1, 0, args.steps + 1)[:-1])
        dummy_key = jax.random.PRNGKey(0)
        return sampling.sample_loop(model, params, dummy_key, latents, steps, 0., sample_step)

    def run_all(weights):
        for i in trange(0, len(weights), args.batch_size):
            outs = run(weights[i:i+args.batch_size])
            for j, out in enumerate(outs):
                utils.to_pil_image(out).save(f'out_{i + j:05}.png')

    try:
        print('Sampling...')
        run_all(jnp.linspace(0, 1, args.n))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
