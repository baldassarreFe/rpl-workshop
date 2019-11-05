FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Common shell utils
RUN apt-get update && \
    apt-get install -y git tree wget htop python3-pip tmux

# Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    sh miniconda.sh -b -p /root/miniconda && \
    /root/miniconda/bin/conda init bash && \
    rm miniconda.sh

# From now on, use bash
SHELL ["bash", "-c"]

# Install project-specific dependencies
COPY conda.yaml /conda.yaml
RUN . "/root/miniconda/etc/profile.d/conda.sh" && \
    conda env update -n base --file /conda.yaml && \
    conda clean --quiet --yes --all
