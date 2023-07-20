import os

import json
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Float32, String

import torch

import numpy as np
import pandas as pd
from scene_classifier_code import SceneClassificationModel
from retinal_sampling.sampling_code import GanglionSampling

class ClassifierROS2Node(Node):
    def __init__(self):
        super().__init__('classifier_node')
        
        # Set the 'use_sim_time' parameter to True
        use_sim_time = Parameter('use_sim_time', value=True)
        self.set_parameters([use_sim_time])

        # Initialize variables
        self.node_time = 0.0
        self.central_time = 0.0
        self.shut_down = False
        self.waiting = False
        self.message = None

        # sampling model
        with open("./retinal_sampling/sampling_parameters.json") as file:
            sampling_parameters = json.load(file)
        self.sampler = GanglionSampling(sampling_parameters)

        # classification model
        self.device = torch.device('cpu')
        with open("./classification_model/model_parameters.json") as file:
            model_parameters = json.load(file)
        self.classmodel = SceneClassificationModel(model_parameters)
        self.classmodel.load_state_dict(torch.load("./classification_model/saved_model.pt", map_location=self.device))
        self.classmodel.eval()
        self.recurrent = self.classmodel.init_recurrent()

        # classification results
        self.classification_results = {'office': [], 'conferenceRoom': [], 'hallway': [], 'auditorium': [], 'openspace': [],
                                       'lobby': [], 'lounge': [], 'pantry': [], 'copyRoom': [], 'storage': [], 'WC': []}

        # publishers
        self.finished_pub = self.create_publisher(Bool, '/finished', 10)

        # subscribers
        self.snapshot_sub = self.create_subscription(Float32, '/camera_node/snapshot', self.snapshot_callback, 10)
        self.waiting_sub = self.create_subscription(Bool, '/camera_node/waiting', self.waiting_callback, 10)
        self.eye_pos_sub = self.create_subscription(Float32, '/saccade_node/eye_pos', self.eye_pos_callback, 10)
        self.shut_down_sub = self.create_subscription(Bool, '/sync_node/shutdown', self.shutdown_callback, 10)
        self.scene_sub = self.create_subscription(String, '/sync_node/scene', self.scene_callback, 10)

        # start the loop
        self.classification_loop()

    # Callback functions
    def snapshot_callback(self, msg):
        resampled = np.clip(self.sampler.resample_image(msg.data) / 127.5 - 1, -1, 1)
        self.classmodel.set_snapshot(resampled)

    def waiting_callback(self, msg):
        self.waiting = msg.data

    def eye_pos_callback(self, msg):
        self.classmodel.set_eye_pos(msg.data)

    def shutdown_callback(self, msg):
        self.shut_down = msg.data

    def scene_callback(self, msg):
        self.scene = msg.data

    # Helper functions
    def get_time(self):
        self.central_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9

    def set_snapshot(self,snapshot):
        if not np.isscalar(snapshot):
            self.snapshot = torch.from_numpy(np.array(snapshot)).permute(0, 3, 1, 2).float().to(self.device)
    
    def set_eye_pos(self,eye_pos):
        self.eye_pos = torch.from_numpy(np.array(eye_pos)).float().to(self.device)

    def update_classification_results(self, class_probability):
        for i, key in enumerate(self.classification_results):
            self.classification_results[key].append(class_probability[i])

    def save_classification_results(self):
        path = os.path.join('/usr/results', self.scene)
        if not os.path.exists(path):
            os.makedirs(path)
            
        file = os.path.join(path, 'classification_results.csv')
        df = pd.DataFrame(self.classification_results)
        df.to_csv(file, index=False)

    # Main loop
    def classification_loop(self):
        while rclpy.ok():
            if self.shut_down:
                # save classification results and shut down
                self.save_classification_results()
                rclpy.shutdown()
                break

            rclpy.spin_once(self)

            self.get_time()
            
            if self.node_time>=self.central_time:
                # wait for next time step
                continue

            if self.waiting:
                # wait for snapshot
                continue

            # Run the classification model on the current snapshot and eye position
            class_probability, self.recurrent = self.classmodel.foward(self.snapshot, self.eye_pos, self.recurrent)
            class_probability = class_probability.detach().numpy()
            
            # Update the classification results
            self.update_classification_results(class_probability)
            
            # Update the node time and publish that the classification is finished
            self.node_time = self.central_time
            self.finished_pub.publish(Bool(data=True))

if __name__ == '__main__':
    rclpy.init()
    classifier_node = ClassifierROS2Node()
