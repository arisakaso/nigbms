# ベースイメージを指定
FROM nvcr.io/nvidia/pytorch:24.03-py3

ARG DEBIAN_FRONTEND=noninteractive

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux \
    htop \
    git \
    wget \
    curl \
    unzip \
    software-properties-common

### fenicsx
RUN add-apt-repository ppa:fenics-packages/fenics
RUN apt-get update && apt-get install -y fenicsx

### OpenFOAM
RUN curl https://dl.openfoam.com/add-debian-repo.sh | bash
RUN apt-get update && apt-get install -y openfoam2312

### required for gmsh
RUN apt-get install libglu1-mesa libxinerama1 libxcursor1 -y

### pip install
RUN pip install --upgrade pip
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install \
    lightning \
    ruff \
    wandb \
    hydra-core \
    tensordict-nightly \
    seaborn \
    jaxtyping \
    torch_optimizer \
    nvitop \
    gmsh \
    fluidfoam \
    h5py \
    ml_collections \
    flax \
    dvc \
    pdoc3 \
    mkdocs

### Enable sudo without password
RUN apt-get install -y sudo 
RUN echo "ALL ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/all && \
    chmod 0440 /etc/sudoers.d/all


# # gitの設定
# ARG GIT_NAME="Sohei Arisaka"
# ARG GIT_EMAIL="sohei.arisaka@u.nus.edu"
# RUN git config --global user.name "${GIT_NAME}" && git config --global user.email "${GIT_EMAIL}"

# 作業ディレクトリを設定
WORKDIR /workspace

# Tunnel 設定
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
RUN tar -xf vscode_cli.tar.gz
RUN rm vscode_cli.tar.gz

# MPI
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/mpi/lib

# # RUN ./code tunnel --accept-server-license-terms