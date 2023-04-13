# Use an official ROS2 base image as the base image
FROM osrf/ros:humble-desktop

LABEL author="Mario Senden"
LABEL email="mario.senden@maastrichtuniversity.nl"

RUN mkdir -p /usr/src/sync_node/results

# Set the working directory
WORKDIR /usr/src/sync_node

# Copy the source code
COPY src/ .

# Start the sync node
RUN chmod +x run.sh
CMD ["./run.sh"]