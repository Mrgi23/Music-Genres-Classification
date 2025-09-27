FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    cmake \
    curl \
    gcc-12 \
    g++-12 \
    git \
    lcov \
    libaubio-dev \
    libcurl4-openssl-dev \
    libomp-dev \
    libzip-dev \
    ninja-build \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    tar && \
    ln -s /usr/bin/python3.12 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/bin:${PATH}"

RUN python3 -m venv /venv && /venv/bin/pip install --upgrade pip

ENV PATH="/venv/bin:${PATH}"

CMD ["/bin/bash"]
