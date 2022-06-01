"""
clone this repo and download the following checkpoints if haven't
git clone https://github.com/kingoflolz/CLIP_JAX
mkdir checkpoints
wget https://v-diffusion.s3.us-west-2.amazonaws.com/wikiart_256.pkl -P checkpoints
wget https://v-diffusion.s3.us-west-2.amazonaws.com/danbooru_128.pkl -P checkpoints
wget https://v-diffusion.s3.us-west-2.amazonaws.com/imagenet_128.pkl -P checkpoints
wget https://v-diffusion.s3.us-west-2.amazonaws.com/wikiart_128.pkl -P checkpoints
"""

import sys
sys.path.insert(0, 'CLIP_JAX')
import tempfile
from pathlib import Path
from functools import partial
from PIL import Image
import jax
import jax.numpy as jnp
import clip_jax
from einops import repeat
from tqdm import tqdm, trange
import cog
from diffusion import get_model, load_params, sampling, utils
from clip_sample import make_normalize, spherical_dist_loss


image_fn, text_fn, clip_params, _ = clip_jax.load('ViT-B/16')
clip_patch_size = 16
clip_size = 224
normalize = make_normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])


class Predictor(cog.Predictor):

    def setup(self):
        print(f'Using device: {jax.devices()}')
        model_names = ['danbooru_128', 'imagenet_128', 'wikiart_128', 'wikiart_256']
        models = {}
        for key in model_names:
            models[key] = get_model(key)
        self.params_zoo = {}
        for key in model_names:
            self.params_zoo[key] = load_params(f'checkpoints/{key}.pkl')
        print('Models loaded!')

    @cog.input(
        "prompt",
        type=str,
        default="a friendly robot, watercolor by James Gurney",
        help="prompt for generating image"
    )
    @cog.input(
        "init_image",
        type=Path,
        default=None,
        help="init image, optional"
    )
    @cog.input(
        "model_name",
        type=str,
        options=['danbooru_128', 'imagenet_128', 'wikiart_128', 'wikiart_256'],
        default='wikiart_256',
        help="model to use"
    )
    @cog.input(
        "iterations",
        type=int,
        default=500,
        help="iterations for generating final image"
    )
    @cog.input(
        "seed",
        type=int,
        default=0,
        help="set to 0 for random seed"
    )
    @cog.input(
        "eta",
        type=float,
        default=1,
        help="the amount of noise to add during sampling (0-1), set to 0 for deterministic (DDIM) sampling, "
             "1 (the default) for stochastic (DDPM) sampling, and in between to interpolate between the two. "
             "DDIM is preferred for low numbers of iterations."
    )
    def predict(self, init_image, prompt, model_name, iterations, seed, eta):
        display_freq = 20
        # some default settings from args
        starting_timestep = 0.9  # the timestep to start at (used with init images)
        batch_size = 1  # the number of images per batch
        clip_guidance_scale = 1000.  # the CLIP guidance scale
        n = 1  # the number of images to sample
        init = None if init_image is None else str(init_image)
        model = get_model(model_name)
        params = self.params_zoo[model_name]
        target_embed = text_fn(clip_params, clip_jax.tokenize([prompt]))

        if init:
            _, y, x = model.shape
            init = Image.open(init).convert('RGB').resize((x, y), Image.LANCZOS)
            init = utils.from_pil_image(init)[None]

        key = jax.random.PRNGKey(seed)
        out_path = Path(tempfile.mkdtemp()) / "out.png"

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
            return grad * -clip_guidance_scale

        def checkin(i, pred, steps, out_path):
            tqdm.write(f'step: {i}, out of {steps} steps')
            utils.to_pil_image(pred).save(str(out_path))
            return out_path

        for i in trange(0, n, batch_size):
            key, subkey = jax.random.split(key)
            cur_batch_size = min(n - i, batch_size)
            # run
            tqdm.write('Sampling...')
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, [cur_batch_size, *model.shape])
            key, subkey = jax.random.split(key)
            cond_params = {'params': params, 'clip_params': clip_params}
            sample_step = partial(sampling.jit_cond_sample_step,
                                  extra_args={},
                                  cond_fn=clip_cond_fn,
                                  cond_params=cond_params)
            steps = utils.get_ddpm_schedule(jnp.linspace(1, 0, iterations + 1)[:-1])
            if init is not None:
                steps = steps[steps < starting_timestep]
                alpha, sigma = utils.t_to_alpha_sigma(steps[0])
                noise = init * alpha + noise * sigma

            for step in trange(len(steps)):
                key, subkey = jax.random.split(subkey)
                if (step + 1) % display_freq == 0 or step == len(steps) - 1:
                    _, pred = sample_step(model, params, subkey, noise, steps[step], steps[step], eta)
                    yield checkin(step + 1, pred[0], iterations, out_path)
                else:
                    noise, _ = sample_step(model, params, subkey, noise, steps[step], steps[step + 1], eta)
