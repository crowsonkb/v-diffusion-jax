# v-diffusion-jax

v objective diffusion inference code for JAX.

The models are denoising diffusion probabilistic models (https://arxiv.org/abs/2006.11239), which are trained to reverse a gradual noising process, allowing the models to generate samples from the learned data distributions starting from random noise. DDIM-style deterministic sampling (https://arxiv.org/abs/2010.02502) is also supported. The models are also trained on continuous timesteps. They use the 'v' objective from Progressive Distillation for Fast Sampling of Diffusion Models (https://openreview.net/forum?id=TIdIXIpzhoI).

## Models:

- Danbooru SFW 128x128

- ImageNet 128x128

- WikiArt 128x128

- WikiArt 256x256

## Sampling

### Unconditional sampling

```
usage: sample.py [-h] [--batch-size BATCH_SIZE] [--checkpoint CHECKPOINT] [--eta ETA] --model
                 {danbooru_128,imagenet_128,wikiart_128,wikiart_256} [-n N] [--seed SEED]
                 [--steps STEPS]
```

`--batch-size`: sample this many images at a time (default 1)

`--checkpoint`: manually specify the model checkpoint file

`--eta`: set to 0 for deterministic (DDIM) sampling, 1 for stochastic (DDPM) sampling, and in between to interpolate between the two. DDIM is preferred for low numbers of timesteps.

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
