# v-diffusion-jax

v objective diffusion inference code for JAX, by Katherine Crowson ([@RiversHaveWings](https://twitter.com/RiversHaveWings)) and Chainbreakers AI ([@jd_pressman](https://twitter.com/jd_pressman)).

The models are denoising diffusion probabilistic models (https://arxiv.org/abs/2006.11239), which are trained to reverse a gradual noising process, allowing the models to generate samples from the learned data distributions starting from random noise. DDIM-style deterministic sampling (https://arxiv.org/abs/2010.02502) is also supported. The models are also trained on continuous timesteps. They use the 'v' objective from Progressive Distillation for Fast Sampling of Diffusion Models (https://openreview.net/forum?id=TIdIXIpzhoI).

## Dependencies

- JAX ([installation instructions](https://github.com/google/jax#installation))

- dm-haiku, einops, numpy, optax, Pillow, tqdm (install with `pip install`)

- CLIP_JAX (https://github.com/kingoflolz/CLIP_JAX), and its additional pip-installable dependencies: ftfy, regex, torch, torchvision (it does not need GPU PyTorch). **If you `git clone --recursive` this repo, it should fetch CLIP_JAX automatically.**

## Model checkpoints:

- [Danbooru SFW 128x128](https://v-diffusion.s3.us-west-2.amazonaws.com/danbooru_128.pkl), SHA-256 `8551fe663dae988e619444efd99995775c7618af2f15ab5d8caf6b123513c334`

- [ImageNet 128x128](https://v-diffusion.s3.us-west-2.amazonaws.com/imagenet_128.pkl), SHA-256 `4fc7c817b9aaa9018c6dbcbf5cd444a42f4a01856b34c49039f57fe48e090530`

- [WikiArt 128x128](https://v-diffusion.s3.us-west-2.amazonaws.com/wikiart_128.pkl), SHA-256 `8fbe4e0206262996ff76d3f82a18dc67d3edd28631d4725e0154b51d00b9f91a`

- [WikiArt 256x256](https://v-diffusion.s3.us-west-2.amazonaws.com/wikiart_256.pkl), SHA-256 `ebc6e77865bbb2d91dad1a0bfb670079c4992684a0e97caa28f784924c3afd81`

## Sampling

### Example

If the model checkpoints are stored in `checkpoints/`, the following will generate an image:

```
./clip_sample.py "a friendly robot, watercolor by James Gurney" --model wikiart_256 --seed 0
```

If they are somewhere else, you need to specify the path to the checkpoint with `--checkpoint`.

### Unconditional sampling

```
usage: sample.py [-h] [--batch-size BATCH_SIZE] [--checkpoint CHECKPOINT] [--eta ETA] --model
                 {danbooru_128,imagenet_128,wikiart_128,wikiart_256} [-n N] [--seed SEED]
                 [--steps STEPS]
```

`--batch-size`: sample this many images at a time (default 1)

`--checkpoint`: manually specify the model checkpoint file

`--eta`: set to 0 for deterministic (DDIM) sampling, 1 (the default) for stochastic (DDPM) sampling, and in between to interpolate between the two. DDIM is preferred for low numbers of timesteps.

`--model`: specify the model to use

`-n`: sample until this many images are sampled (default 1)

`--seed`: specify the random seed (default 0)

`--steps`: specify the number of diffusion timesteps (default is 1000, can lower for faster but lower quality sampling)

### CLIP guided sampling

CLIP guided sampling lets you generate images with diffusion models conditional on the output matching a text prompt.

```
usage: clip_sample.py [-h] [--batch-size BATCH_SIZE] [--checkpoint CHECKPOINT]
                      [--clip-guidance-scale CLIP_GUIDANCE_SCALE] [--eta ETA] --model
                      {danbooru_128,imagenet_128,wikiart_128,wikiart_256} [-n N] [--seed SEED]
                      [--steps STEPS]
                      prompt
```

`clip_sample.py` has the same options as `sample.py` and these additional ones:

`prompt`: the text prompt to use

`--clip-guidance-scale`: how strongly the result should match the text prompt (default 1000)
