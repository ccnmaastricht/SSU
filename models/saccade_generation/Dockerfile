# Use an official ROS2 base image as the base image
FROM osrf/ros:foxy-desktop

LABEL author="Mario Senden"
LABEL email="mario.senden@maastrichtuniversity.nl"

# Update package list and Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake libreadline6-dev libncurses5-dev libgsl0-dev \
    python3-dev python3-numpy python3-scipy python3-matplotlib ipython3 python3-nose python3-pip wget

# Install cython using pip
RUN pip3 install cython

# Download NEST
WORKDIR /root
RUN wget https://github.com/nest/nest-simulator/archive/v2.18.0.tar.gz
RUN tar -xvzf v2.18.0.tar.gz
WORKDIR /root/nest-simulator-2.18.0

# Build and install NEST
RUN mkdir build
WORKDIR /root/nest-simulator-2.18.0/build
RUN cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr ..
RUN make
RUN make install

WORKDIR /root
RUN rm -rf nest-simulator-2.18.0 v2.18.0.tar.gz

# Add NEST to the PYTHONPATH
ENV PYTHONPATH $PYTHONPATH:/usr/lib/x86_64-linux-gnu/python3.10/site-packages

# Set the working directory
WORKDIR /usr/src

# Copy the requirements file
COPY requirements.txt .

# Install the required packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ .

# Start the saccade generation node
RUN chmod +x run.sh
CMD ["./run.sh"]