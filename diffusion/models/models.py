import pickle

import jax
import jax.numpy as jnp

from . import danbooru_128, imagenet_128, wikiart_128, wikiart_256


models = {
    'danbooru_128': danbooru_128.Danbooru128Model,
    'imagenet_128': imagenet_128.ImageNet128Model,
    'wikiart_128': wikiart_128.WikiArt128Model,
    'wikiart_256': wikiart_256.WikiArt256Model,
}


def get_model(model):
    return models[model]


def get_models():
    return list(models.keys())


def load_params(checkpoint):
    with open(checkpoint, 'rb') as fp:
        return jax.tree_map(jnp.array, pickle.load(fp)['params_ema'])
