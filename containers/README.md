# Containerized DiT for HPC


## Build

The containerized DiT can be built with `cotainr`, a convenient utility tool to build Apptainer images with support for conda environments.

```shell
cotainr build photocast-dit.sif -vvv --base-image=docker://nvidia/cuda:12.6.2-base-ubuntu22.04 --conda-env=environment.yml
```

## Run

A Slurm `exec` script is provided to train the model with `torchrun`. The job can be submitted with the following command:

```shell
sbatch ./containers/dit_slurm.sh
```