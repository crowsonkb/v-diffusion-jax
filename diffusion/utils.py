import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


def from_pil_image(x):
    """Converts from a PIL image to a JAX array."""
    x = jnp.array(x)
    if x.ndim == 2:
        x = x[..., None]
    return x.transpose((2, 0, 1)) / 127.5 - 1


def to_pil_image(x):
    """Converts from a JAX array to a PIL image."""
    if x.ndim == 4:
        assert x.shape[0] == 1
        x = x[0]
    if x.shape[0] == 1:
        x = x[0]
    else:
        x = x.transpose((1, 2, 0))
    arr = np.array(jnp.round(jnp.clip((x + 1) * 127.5, 0, 255)).astype(jnp.uint8))
    return Image.fromarray(arr)


def log_snr_to_alpha_sigma(log_snr):
    """Returns the scaling factors for the clean image and for the noise, given
    the log SNR for a timestep."""
    return jnp.sqrt(jax.nn.sigmoid(log_snr)), jnp.sqrt(jax.nn.sigmoid(-log_snr))


def alpha_sigma_to_log_snr(alpha, sigma):
    """Returns a log snr, given the scaling factors for the clean image and for
    the noise."""
    return jnp.log(alpha**2 / sigma**2)


def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean image and for the noise, given
    a timestep."""
    return jnp.cos(t * jnp.pi / 2), jnp.sin(t * jnp.pi / 2)


def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return jnp.arctan2(sigma, alpha) / jnp.pi * 2


def get_ddpm_schedule(ddpm_t):
    """Returns timesteps for the noise schedule from the DDPM paper."""
    log_snr = -jnp.log(jnp.expm1(1e-4 + 10 * ddpm_t**2))
    alpha, sigma = log_snr_to_alpha_sigma(log_snr)
    return alpha_sigma_to_t(alpha, sigma)
