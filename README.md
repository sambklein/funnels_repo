# Funnels

This repository contains the code needed to reproduce the experiments from the paper:


> Funnels: Exact maximum likelihood with dimensionality reduction.

## Dependencies

The base container for this project was pulled from the docker registry with the
tag `pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime`. See `docker/requirements.txt` for the necessary pip packages to be
installed into the container.

## Usage

The environment variable REPOROOT must be set to point to the top level of the repository. Once the container has been
built or pulled run the following:

```angular2html
chmod +x run_setup.sh
./run_setup.sh 
```

and everything should run. All experiments ran this container using singularity 3.4.0.

## Data

The [preprocessed datasets for MAF experiments](https://zenodo.org/record/1161203#.YaomofHMKji) was used for the density
estimation comparisons.

## Plane data experiments

Run `experiments/plane_data_generation.py`.

## Tabular data experiments

Run `experiments/uci.py`.

## Image experiments

All image experiments have the args saved in json and are launched using `experiments/image_generation.py`, with
cifar-10 and imagenet funnel experiments run with `--model 'funnel_conv'`.

The Inception and FID scores were calculated using external libraries as described in the directories under `external/`.

## Anomaly detection experiments

For the anomaly detection experiments the defaults can be found in `experiments/image_configs/AD_config.json`. With
these defaults set across experiments the different models can be run as follows:

VAE

```angular2html
python image_generation.py --model 'VAE' --latent_size 4
python image_generation.py --model 'VAE' --latent_size 16
```

F-NSF

```angular2html
python image_generation.py --model 'funnel_conv_deeper' --latent_size 4
python image_generation.py --model 'funnel_conv_deeper' --latent_size 16
```

F-MLP

```angular2html
python image_generation.py --model 'funnelMLP' --levels 4
python image_generation.py --model 'funnelMLP' --levels 3
```

NSF

```angular2html
python image_generation.py --model 'glow'
```