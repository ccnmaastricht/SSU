#!/bin/bash

# Trap signals to shut down the Docker container
trap "exit" SIGINT SIGTERM

# Activate the NEST environment
source $CONDA_DIR/bin/activate nest_env

# Start the ROS node
python saccade_ros2_node.py &

# Store the PID of the ROS node
pid=$!

# Wait for the ROS node to finish
wait $pid

# Stop the Docker container once the ROS node finishes
kill -s SIGINT 1