from einops import repeat
import jax
import jax.numpy as jnp
from tqdm import trange

from . import utils


def sample_step(model, params, key, x, t, t_next, eta, extra_args):
    dummy_key = jax.random.PRNGKey(0)
    v = model.apply(params, dummy_key, x, repeat(t, '-> n', n=x.shape[0]), extra_args)
    alpha, sigma = utils.t_to_alpha_sigma(t)
    key, subkey = jax.random.split(key)
    pred = x * alpha - v * sigma
    eps = x * sigma + v * alpha
    alpha_next, sigma_next = utils.t_to_alpha_sigma(t_next)
    ddim_sigma = eta * jnp.sqrt(sigma_next**2 / sigma**2) * \
        jnp.sqrt(1 - alpha**2 / alpha_next**2)
    adjusted_sigma = jnp.sqrt(sigma_next**2 - ddim_sigma**2)
    x = pred * alpha_next + eps * adjusted_sigma
    x = x + jax.random.normal(key, x.shape) * ddim_sigma
    return x, pred


jit_sample_step = jax.jit(sample_step, static_argnums=0)


def cond_sample_step(model, params, key, x, t, t_next, eta, extra_args, cond_fn, cond_params):
    dummy_key = jax.random.PRNGKey(0)
    v = model.apply(params, dummy_key, x, repeat(t, '-> n', n=x.shape[0]), extra_args)
    alpha, sigma = utils.t_to_alpha_sigma(t)
    key, subkey = jax.random.split(key)
    cond_grad = cond_fn(x, subkey, t, extra_args, **cond_params)
    v = v - cond_grad * (sigma / alpha)
    pred = x * alpha - v * sigma
    eps = x * sigma + v * alpha
    alpha_next, sigma_next = utils.t_to_alpha_sigma(t_next)
    ddim_sigma = eta * jnp.sqrt(sigma_next**2 / sigma**2) * \
        jnp.sqrt(1 - alpha**2 / alpha_next**2)
    adjusted_sigma = jnp.sqrt(sigma_next**2 - ddim_sigma**2)
    x = pred * alpha_next + eps * adjusted_sigma
    x = x + jax.random.normal(key, x.shape) * ddim_sigma
    return x, pred


jit_cond_sample_step = jax.jit(cond_sample_step, static_argnums=(0, 8))


def sample_loop(model, params, key, x, steps, eta, sample_step):
    for i in trange(len(steps)):
        key, subkey = jax.random.split(key)
        if i < len(steps) - 1:
            x, _ = sample_step(model, params, subkey, x, steps[i], steps[i + 1], eta)
        else:
            _, pred = sample_step(model, params, subkey, x, steps[i], steps[i], eta)
    return pred


def reverse_sample_step(model, params, key, x, t, t_next, extra_args):
    dummy_key = jax.random.PRNGKey(0)
    v = model.apply(params, dummy_key, x, repeat(t, '-> n', n=x.shape[0]), extra_args)
    alpha, sigma = utils.t_to_alpha_sigma(t)
    pred = x * alpha - v * sigma
    eps = x * sigma + v * alpha
    alpha_next, sigma_next = utils.t_to_alpha_sigma(t_next)
    x = pred * alpha_next + eps * sigma_next
    return x, pred


jit_reverse_sample_step = jax.jit(reverse_sample_step, static_argnums=0)


def reverse_sample_loop(model, params, key, x, steps, sample_step):
    for i in trange(len(steps) - 1):
        key, subkey = jax.random.split(key)
        x, _ = sample_step(model, params, subkey, x, steps[i], steps[i + 1])
    return x
