# Base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Avoid user interaction
ENV DEBIAN_FRONTEND=noninteractive

# Environment Variables
ENV CONDA_URL=https://github.com/conda-forge/miniforge/releases/download/23.3.1-1/Miniforge3-23.3.1-1-Linux-x86_64.sh
ENV CONDA_INSTALL_PATH=/opt/conda
ENV DEPENDENCIES_DIR=/tmp/dependencies

# Copy needed files
COPY ./docker/dependencies ${DEPENDENCIES_DIR}

# Install apt dependencies
RUN apt-get update && \
    xargs -a ${DEPENDENCIES_DIR}/apt-runtime.txt apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Set timezone and locale
RUN echo "Etc/UTC" > /etc/timezone && \
    ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install Miniforge (Conda)
RUN mkdir -p /tmp/conda && \
    curl -fvL -o /tmp/conda/miniconda.sh ${CONDA_URL} && \
    bash /tmp/conda/miniconda.sh -b -p ${CONDA_INSTALL_PATH} -u && \
    rm -rf /tmp/conda
# make mamba visible (using mamba for efficiency)
ENV PATH=${CONDA_INSTALL_PATH}/condabin:${CONDA_INSTALL_PATH}/bin:${PATH}

# Install Mamba for faster Conda operations
RUN conda install mamba -n base -c conda-forge

# Create Conda environment
RUN mamba env create --file ${DEPENDENCIES_DIR}/environment.yaml

# Activate Conda environment
ENV PATH=${CONDA_INSTALL_PATH}/envs/defog/bin:${PATH}

# Install additional pip packages without dependencies if needed
RUN chmod +x ${DEPENDENCIES_DIR}/pip_no_deps.sh && \
    ${DEPENDENCIES_DIR}/pip_no_deps.sh

# Clean up
RUN mamba clean -a -y && \
    find ${CONDA_INSTALL_PATH}/envs/defog -name '__pycache__' -type d -exec rm -rf {} +

# Initialize Conda for shells
RUN mamba init --system bash && \
    echo "mamba activate defog" >> /etc/profile.d/conda.sh

# Delete temporary files
RUN rm -rf ${DEPENDENCIES_DIR}

# Set working directory
WORKDIR /workspace

# Entrypoint script
COPY ./docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["bash"]
