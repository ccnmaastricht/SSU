# Use an official ROS2 base image as the base image
FROM osrf/ros:foxy-desktop

LABEL author="Mario Senden"
LABEL email="mario.senden@maastrichtuniversity.nl"

# Create the data directory
RUN mkdir -p /usr/results

# Set the working directory
WORKDIR /usr/src

# Copy the requirements file
COPY requirements.txt .

# Install the required packages
RUN apt-get update && apt-get install -y python3-pip \
    && pip3 install --no-cache-dir -r requirements.txt
    
# Install PyTorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy the source code
COPY src/ .

# Start the scene classification node
RUN chmod +x run.sh
CMD ["./run.sh"]