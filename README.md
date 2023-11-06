# SSU Project
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
4. **Time Synchronization with a Simulation Coordinator**: The Simulation Coordinator ensures that all modules are in sync, particularly in terms of simulated time.
5. **Synchronization of Components**: Each module signals when it is done with its calculations for the current time slot, and the Simulation Coordinator starts a new one once all modules are ready.

In the SSU project, these steps are realized as follows:

- **Component Modularization**: The system is divided into various modules such as scene classification, saliency computation, saccade generation, etc.
- **Containerization**: Each of these modules is containerized using Docker.
- **Communication through Message Broker**: The modules communicate with each other using ROS2 as the message broker.
- **Time Synchronization with a Simulation Coordinator**: A dedicated `sync_node` serves as the Simulation Coordinator to keep all simulations in sync.
- **Synchronization of Components**: Each module sends a message once it completes its calculations for a cycle, signaling the Simulation Coordinator to start a new cycle.

## SSU Architecture

1. **Camera Module**: This module simulates a virtual camera. It uses panorama images of different scenes to simulate the visual input that would be received by an eye. It provides a 360-degree view of each scene and can capture images based on the current eye position and target location. The module uses Docker to ensure that its dependencies are self-contained and that it can run in a variety of environments. It contains several Python scripts that provide the camera functionality and set up a ROS2 node for communication with other modules.

2. **Saccade Generation Module**: This module simulates the biological process of saccades, which are rapid eye movements that quickly redirect the focus of the eye to a new position. The module uses NEST, a simulator for spiking neural network models, to construct and simulate a Saccade Generator. The module also includes a ROS2 node for communication with other modules. 

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

This directory contains files for the 'sync_node', a special ROS2 node that acts as the Simulation Coordinator for the entire system, ensuring that all modules are synchronized with respect to simulated time. 

## Dataset Requirement

To successfully run the SSU system, the following requirements must be met:

- **2D-3D-S Dataset**: You must obtain at least one RGB image from each scene class from the [2D-3D-S dataset](https://github.com/alexsax/2D-3D-Semantics). 
- **Directory Structure**: An image must be placed within `./models/camera/data/CLASS_NAME` folder and named `image.png`.

The 11 classes are:
- 'hallway'
- 'lounge'
- 'storage'
- 'copyRoom'
- 'openspace'
- 'auditorium'
- 'pantry'
- 'conferenceRoom'
- 'office'
- 'lobby'
- 'WC'

## Additional Information

The 2D-3D-S dataset was also utilized to train the scene classifier model. A csv file, `LABEL_holdout.csv`, contains the path and filename of all images that were not used during training. This list is available in the `./models/camera/data` folder.

## Quick Start

To get started with running the system:

1. Ensure that Docker and Docker Compose are installed on your system.

2. Clone the project repository.

3. Provide the scene (class) you would like the system to explore in the `simulation_configuration.json` file within the `./config` folder.

4. Place an image named `image.png` of that scene (class) in the `./models/camera/data/CLASS_NAME` folder.

5. Use the `build_containers.sh` script to build the Docker images for all nodes. 

```
./build_containers.sh
```

6. Use Docker Compose to run the system:

```
docker-compose up
```

To stop the system, press `CTRL+C` in the terminal. This will stop all the running containers.


## Contributors
 
The SSU architecture was implemented by Mario Senden<sup>a, b</sup>. 

The Saccade Generation module is adapted from [spiking neural network model of the saccade generator in the reticular formation](https://github.com/ccnmaastricht/spiking_saccade_generator), written by Anno Kurth<sup>c, d, e</sup> with support from Sacha van Albada<sup>c, d, e, f</sup>. The spiking neural network model is itself inspired by work of Gancarz and Grossberg [1]. 

The Saliency module utilizes the [contextual encoder-decoder network for visual saliency prediction](https://github.com/alexanderkroner/saliency) developed by Alexander Kroner<sup>a, b</sup> [2]. 

The Scene Classification module utilizes a [retinal sampling procedure](https://github.com/ccnmaastricht/ganglion_cell_sampling) written by Danny da Costa<sup>a, b</sup> [3].  

**a)** Department of Cognitive Neuroscience, Faculty of Psychology and Neuroscience, Maastricht University, Maastricht, The Netherlands

**b)** Maastricht Brain Imaging Centre, Faculty of Psychology and Neuroscience, Maastricht University, Maastricht, The Netherlands

**c)** Institute of Neuroscience and Medicine (INM-6), Jülich Research Centre, Jülich, Germany

**d)** Institute for Advanced Simulation (IAS-6), Jülich Research Centre, Jülich, Germany

**e)** ARA-Institute Brain Structure-Function Relationships (INM-10), Jülich Research Centre, 	Jülich, Germany

**f)** Institute of Zoology, Faculty of Mathematics and Natural Sciences, University of Cologne, Cologne, Germany

## License

This repository contains a mix of open-source code under two different licenses:

- **MIT License** With the exception of the Saccade Generation Module, this project is under the MIT license. A copy of the MIT license can be found in the LICENSE file.
- **CC BY-NC-SA 4.0 License**: The Saccade Generation Module is based on the [spiking neural network model of the saccade generator in the reticular formation](https://github.com/ccnmaastricht/spiking_saccade_generator) written by Anno Kurth with support from Sacha van Albada and is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (CC BY-NC-SA 4.0). The full text of the CC BY-NC-SA 4.0 License can be found in the [CC_BY_NC_SA_LICENSE.md](./CC_BY_NC_SA_LICENSE.md) file.

## References
[1] Gancarz, Gregory, and Stephen Grossberg. "A neural model of the saccade generator in the reticular formation." Neural Networks 11.7-8 (1998): 1159-1174. 

[2] Kroner, A., Senden, M., Driessens, K., & Goebel, R. (2020). Contextual encoder–decoder network for visual saliency prediction. Neural Networks, 129, 261-270.

[3] da Costa, D., Kornemann, L., Goebel, R., & Senden, M. (2023). Unlocking the Secrets of the Primate Visual Cortex: A CNN-Based Approach Traces the Origins of Major Organizational Principles to Retinal Sampling. bioRxiv, 2023-04.
