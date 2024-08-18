# ベースイメージを指定
FROM nvcr.io/nvidia/pytorch:24.07-py3
# FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime # this does not work


ARG DEBIAN_FRONTEND=noninteractive

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux \
    htop \
    git \
    wget \
    curl \
    unzip \
    software-properties-common \
    lsof

### required for gmsh
RUN apt-get install libglu1-mesa libxinerama1 libxcursor1 -y

### OpenFOAM
RUN curl https://dl.openfoam.com/add-debian-repo.sh | bash
RUN apt-get update && apt-get install -y openfoam2312

### fenicsx
RUN apt-get install -y gpg-agent
RUN add-apt-repository ppa:fenics-packages/fenics
RUN apt-get update && apt-get install -y fenicsx


### pip install
RUN pip install --upgrade pip
RUN pip install \
    tensordict-nightly \    
    lightning \
    ruff \
    wandb \
    hydra-core \
    seaborn \
    jaxtyping \
    torch_optimizer \
    nvitop \
    gmsh \
    fluidfoam \
    h5py \
    ml_collections \
    dvc \
    pydevd \
    pdoc3 \
    mkdocs \
    mkdocs-material \
    mkdocs-gen-files \
    mkdocs-literate-nav \
    mkdocstrings \
    mkdocstrings-python


# MPI
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/mpi/lib:/opt/hpcx/ucx/lib