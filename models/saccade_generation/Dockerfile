# Use an official ROS2 base image as the base image
FROM osrf/ros:humble-desktop

LABEL author="Mario Senden"
LABEL email="mario.senden@maastrichtuniversity.nl"

# Install wget
RUN apt-get update && apt-get install wget

# Set environment variables to control the Miniconda installation.
ENV MINICONDA_VERSION latest
ENV CONDA_DIR $HOME/miniconda3

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Add Miniconda to PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Update Conda and create NEST environment
RUN conda update -y conda \
    && conda create --name nest_env -c conda-forge nest-simulator=2.18.0 python

# Change the default shell to be within the nest_env Conda environment
SHELL ["conda", "run", "-n", "nest_env", "/bin/bash", "-c"]

# Set the working directory
WORKDIR /usr/src

# Copy the requirements file
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ .

# Start the saccade generation node
RUN chmod +x run.sh
CMD ["./run.sh"]