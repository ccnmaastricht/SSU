# Use an argument to control whether the CPU or GPU version of TensorFlow is installed (default is CPU version 2.7.0)
ARG tensorflow_version="tensorflow==2.7.0"

# Use an official ROS2 base image as the base image
FROM osrf/ros:humble-desktop

LABEL author="Mario Senden"
LABEL email="mario.senden@maastrichtuniversity.nl"

# Set the working directory
WORKDIR /usr/src/saliency

# Copy the requirements file
COPY requirements.txt .

# Install the required packages & clean up the package list to reduce the image size.
RUN apt-get update && apt-get install -y python3-pip \
    && pip3 install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/* 

# Install NVIDIA's CUDA Toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin; \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600; \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub; \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"; \
    apt-get update; \
    apt-get -y install cuda-toolkit-11-0; \
    rm -rf /var/lib/apt/lists/*


# Install TensorFlow
RUN pip3 install --upgrade pip && \
    pip3 install ${tensorflow_version}

# Copy the source code
COPY src/ .

# Start the saliency node
RUN chmod +x run.sh
CMD ["./run.sh"]