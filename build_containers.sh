#!/bin/bash

# build all containers
docker build -t camera_node -f models/camera/Dockerfile models/camera
docker build -t classifier_node -f models/scene_classification/Dockerfile models/scene_classification
docker build -t saccade_node -f models/saccade_generation/Dockerfile models/saccade_generation
docker build -t saliency_node -f models/saliency/Dockerfile models/saliency
docker build -t selection_node -f models/target_selection/Dockerfile models/target_selection
docker build -t sync_node -f sync_node/Dockerfile sync_node