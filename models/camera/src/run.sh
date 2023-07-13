#!/bin/bash

# Trap signals to shut down the Docker container
trap "exit" SIGINT SIGTERM

# Start the ROS node
python3 camera_ros2_node.py &

# Store the PID of the ROS node
pid=$!

# Wait for the ROS node to finish
wait $pid

# Stop the Docker container once the ROS node finishes
kill -s SIGINT 1