FROM ubuntu:20.04 AS base

# Fix timezone issue
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /work

# install anaconda (from continuumIO)
FROM base as anaconda
ENV CONDA_VERSION 2022.05
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# hadolint ignore=DL3008
RUN set -x && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        git \
        libglib2.0-0 \
        libsm6 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
        libxrender1 \
        mercurial \
        openssh-client \
        procps \
        subversion \
        wget \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* && \
    UNAME_M="$(uname -m)" && \
    if [ "${UNAME_M}" = "x86_64" ]; then \
        ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh"; \
        SHA256SUM="a7c0afe862f6ea19a596801fc138bde0463abcbce1b753e8d5c474b506a2db2d"; \
    elif [ "${UNAME_M}" = "s390x" ]; then \
        ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-${CONDA_VERSION}-Linux-s390x.sh"; \
        SHA256SUM="c14415df69e439acd7458737a84a45c6067376cbec2fccf5e2393f9837760ea7"; \
    elif [ "${UNAME_M}" = "aarch64" ]; then \
        ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-${CONDA_VERSION}-Linux-aarch64.sh"; \
        SHA256SUM="dc6bb4eab3996e0658f8bc4bbd229c18f55269badd74acc36d9e23143268b795"; \
    elif [ "${UNAME_M}" = "ppc64le" ]; then \
        ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-${CONDA_VERSION}-Linux-ppc64le.sh"; \
        SHA256SUM="a50bf5bd26b5c5a2c24028c1aff6da2fa4d4586ca43ae3acdf7ffb9b50d7f282"; \
    fi && \
    wget "${ANACONDA_URL}" -O anaconda.sh -q && \
    echo "${SHA256SUM} anaconda.sh" > shasum && \
    sha256sum --check --status shasum && \
    /bin/bash anaconda.sh -b -p /opt/conda && \
    rm anaconda.sh shasum && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# Install Latex
FROM anaconda as latex

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    texlive-full
RUN apt-get install -y biber
# set env
# ENV PATH=/usr/local/texlive/2018/bin/x86_64-linux:$PATH

# install cuda packages
FROM latex as cudnn
ENV OS=ubuntu2004

RUN apt clean && apt update && apt upgrade && apt-get install -y software-properties-common

RUN apt-get install -y gnupg &&\
    wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin && \
    mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/3bf863cc.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /" && \
    apt-get update

ENV cudnn_version=8.5.0.96
ENV cuda_version=cuda11.7

RUN apt-get install libcudnn8=${cudnn_version}-1+${cuda_version} && \
    apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}

# setup ML environment
FROM cudnn as ml_env
COPY environment.yml environment.yml
RUN conda env create --file environment.yml

WORKDIR /work