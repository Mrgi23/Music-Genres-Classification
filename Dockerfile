# Use Ubuntu 22.04 which includes Python 3.10
FROM ubuntu:22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update apt and install required packages:
# - cmake, clang, llvm, git for building
# - libcurl4-openssl-dev, libopenblas-dev, liblapack-dev, libzip-dev for execution
# - python3.10 and related packages for Python
# - lcov for coverage tools
RUN apt update && apt install -y \
    cmake \
    clang \
    curl \
    git \
    lcov \
    libaubio-dev \
    libcurl4-openssl-dev \
    libzip-dev \
    llvm \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    tar && \
    rm -rf /var/lib/apt/lists/*

# Set PATH to include /usr/local/bin
ENV PATH="/usr/local/bin:${PATH}"

# Create virtual environment
RUN python3 -m venv /venv && /venv/bin/pip install --upgrade pip

# Add the virtual environment's bin directory to PATH.
ENV PATH="/venv/bin:${PATH}"

# By default, run a shell.
CMD ["/bin/bash"]
