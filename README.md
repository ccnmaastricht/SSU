# SSU Project

Welcome to the Scene Understanding with Saccades (SSU) project. This is a tech demo that showcases the power and versatility of the modular-integrative modeling approach. 

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
