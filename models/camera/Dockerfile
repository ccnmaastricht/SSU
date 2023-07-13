# Use an official ROS2 base image as the base image
FROM osrf/ros:humble-desktop

LABEL author="Mario Senden"
LABEL email="mario.senden@maastrichtuniversity.nl"

# Set the working directory
WORKDIR /usr/src

# Create the data directory
RUN mkdir -p /usr/data

# Copy the requirements file
COPY requirements.txt .

# Install the required packages
RUN apt-get update && apt-get install -y python3-pip \
    && pip3 install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ .

# Copy the data
COPY data/ /usr/data

# Start the camera node
RUN chmod +x run.sh
CMD ["./run.sh"]