# SSU Project
![](https://img.shields.io/github/license/ccnmaastricht/SSU)
![](https://img.shields.io/github/issues/ccnmaastricht/SSU)
![](https://img.shields.io/github/forks/ccnmaastricht/SSU)
![](https://img.shields.io/github/stars/ccnmaastricht/SSU)

Welcome to the Saccades for Scene Understanding (SSU) project. This is a tech demo that showcases the power and versatility of the modular-integrative modeling approach. 

## Introduction
With an increasing need for complex modeling and simulations in neuroscience, it is essential to have an approach that allows for flexibility, scalability, and efficient management. The modular-integrative modeling approach (advanced by work package 3 of the Human Brain Project's 3rd Specific Grant Agreement) meets these needs by dividing the entire model into smaller, manageable modules. This allows for components to be developed independently, and then integrated and synchronized to simulate the complete system. 

In this project, we have chosen to showcase this approach through a high level functional capacity that requires the coordinated interplay between several modules - Saccades for Scene Understanding (SSU). Scene understanding involves several capacities including scene classification, navigation, identification and localization of objects, among others. For the purpose of this demo, we focus on scene classification. 

## Why Scene Understanding with Saccades?
Scene understanding is a complex task requiring the interplay of numerous modules - each processing a different aspect of visual input, coordinating eye movements, decision making and more. It offers an ideal test bed for a modular-integrative approach due to the following reasons:
1. **Complexity**: The multiple sub-tasks involved provide the chance to showcase different modules.
2. **Interactivity**: The need for the modules to coordinate and communicate clearly demonstrates the integrative part of this approach.

## Modular-Integrative Modeling Approach
The modular-integrative modeling approach involves several steps:

1. **Component Modularization**: The system is broken down into smaller components or modules, each of which can be developed independently.
2. **Containerization**: Each module is containerized using technologies like Docker, to avoid dependency conflicts and ensure isolated environments.
3. **Communication through Message Broker**: Modules exchange data via a message broker using a publish-subscribe pattern.
4. **Time Synchronization with a Simulation Manager**: The Simulation Manager ensures that all modules are in sync, particularly in terms of simulated time.
5. **Synchronization of Components**: Each module signals when it is done with its calculations for the current time slot, and the Simulation Manager starts a new one once all modules are ready.

In the SSU project, these steps are realized as follows:

- **Component Modularization**: The system is divided into various modules such as scene classification, saliency computation, saccade generation, etc.
- **Containerization**: Each of these modules is containerized using Docker.
- **Communication through Message Broker**: The modules communicate with each other using ROS2 as the message broker.
- **Time Synchronization with a Simulation Manager**: A dedicated `sync_node` serves as the Simulation Manager to keep all simulations in sync.
- **Synchronization of Components**: Each module sends a message once it completes its calculations for an epoch, signaling the Simulation Manager to start a new epoch.

## SSU Architecture

1. **Camera Module**: This module simulates a virtual camera. It uses panorama images of different scenes to simulate the visual input that would be received by an eye. It provides a 360-degree view of each scene and can capture images based on the current eye position and target location. The module uses Docker to ensure that its dependencies are self-contained and that it can run in a variety of environments. It contains several Python scripts that provide the camera functionality and set up a ROS2 node for communication with other modules.

2. **Saccade Generation Module**: This module simulates the biological process of saccades, which are rapid eye movements that quickly redirect the focus of the eye to a new position. The module uses the Nest library, a simulator for spiking neural network models, to construct and simulate a Saccade Generator. The module also includes a ROS2 node for communication with other modules. 

3. **Saliency Module**: The purpose of the saliency module is to simulate the neural processes that highlight regions of an image that are likely to draw attention. It uses a TensorFlow-based saliency model to generate a saliency map from an input image. The module includes a ROS2 node for communication with other modules.

4. **Scene Classification Module**: This module simulates the neural processes that recognize and classify different types of scenes based on visual input. It uses a neural network model for scene classification, taking an image as input and outputting a prediction of the scene class. The module includes a ROS2 node for communication with other modules.

5. **Target Selection Module**: The purpose of this module is to select a target location in a scene based on a saliency map.

6. **Sync Node**: This is the central node that synchronizes the operation of all other nodes. It manages the overall simulation, including advancing simulated time and orchestrating the shutdown process when the simulation is complete.


## Project Structure

```
.
├── build_containers.sh
├── config
│   └── simulation_configuration.json
├── docker-compose.yml
├── models
│   ├── camera
│   │   ├── data
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── src
│   ├── saccade_generation
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── src
│   ├── saliency
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── src
│   ├── scene_classification
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── src
│   └── target_selection
│       ├── Dockerfile
│       ├── requirements.txt
│       └── src
├── results
└── sync_node
    ├── Dockerfile
    └── src
        ├── run.sh
        └── sync_node.py
```

### `build_containers.sh`

A shell script that is used to build Docker containers for each node. 

### `docker-compose.yml`

A Docker Compose file that is used to manage and run the containers as a single, integrated service. 

### `models`

This directory contains the different modules (or "models") of the system, each implemented as a ROS2 node. Each model has its own Dockerfile, a requirements.txt file listing the Python dependencies, and a 'src' directory containing the source code.

The 'src' directories typically contain a Python script defining a ROS2 node for the module, a Python script containing the core functionality of the module, and a shell script ('run.sh') for running the node in a Docker container. 

The 'models' directory also contains sub-directories for data or additional resources needed by the modules, like neural network model files, images, parameter files, etc.

### `sync_node`

This directory contains files for the 'sync_node', a special ROS2 node that acts as the Simulation Manager for the entire system, ensuring that all modules are synchronized with respect to simulated time. 

## Quick Start

To get started with running the system:

1. Ensure that Docker and Docker Compose are installed on your system.

2. Clone the project repository.

3. Use the `build_containers.sh` script to build the Docker images for all nodes. 

```
./build_containers.sh
```

4. Use Docker Compose to run the system:

```
docker-compose up
```

To stop the system, press `CTRL+C` in the terminal. This will stop all the running containers.
