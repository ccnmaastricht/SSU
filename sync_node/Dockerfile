# Use an official ROS2 base image as the base image
FROM osrf/ros:humble-desktop

LABEL author="Mario Senden"
LABEL email="mario.senden@maastrichtuniversity.nl"

# Set the working directory
WORKDIR /usr/src

# Create the configuration directory
RUN mkdir -p /usr/config

# Copy the source code
COPY src/ .

# Start the sync node
RUN chmod +x run.sh
CMD ["./run.sh"]