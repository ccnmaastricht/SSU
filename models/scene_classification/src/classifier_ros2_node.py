import os

import json
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Int32, Float32MultiArray, String

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
        self.node_id = 4
        self.eye_pos = torch.zeros((1,2))


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
        self.snapshot = torch.zeros((1, 3, model_parameters['image_size'], model_parameters['image_size'])).to(self.device)

        # classification results
        self.classification_results = {'office': [], 'conferenceRoom': [], 'hallway': [], 'auditorium': [], 'openspace': [],
                                       'lobby': [], 'lounge': [], 'pantry': [], 'copyRoom': [], 'storage': [], 'WC': []}

        # publishers
        self.finished_pub = self.create_publisher(Int32, '/finished', 10)

        # subscribers
        self.snapshot_sub = self.create_subscription(Float32MultiArray, '/camera_node/snapshot', self.snapshot_callback, 10)
        self.waiting_sub = self.create_subscription(Bool, '/camera_node/waiting', self.waiting_callback, 10)
        self.eye_pos_sub = self.create_subscription(Float32MultiArray, '/saccade_node/eye_pos', self.eye_pos_callback, 10)
        self.shut_down_sub = self.create_subscription(Bool, '/sync_node/shutdown', self.shutdown_callback, 10)
        self.scene_sub = self.create_subscription(String, '/sync_node/scene', self.scene_callback, 10)

        # start the loop
        self.classification_loop()

    # Callback functions
    def snapshot_callback(self, msg):
        # Convert the data field to a NumPy array and reshape it to its original shape
        snapshot = np.array(msg.data)
        snapshot = snapshot.reshape((msg.layout.dim[0].size, msg.layout.dim[1].size, msg.layout.dim[2].size))

        # Resample the snapshot and pass it to the classification model
        resampled = np.clip(self.sampler.resample_image(snapshot) / 127.5 - 1, -1, 1)
        self.set_snapshot(resampled)

    def waiting_callback(self, msg):
        self.waiting = msg.data

    def eye_pos_callback(self, msg):
        self.set_eye_pos(msg.data)

    def shutdown_callback(self, msg):
        self.shut_down = msg.data

    def scene_callback(self, msg):
        self.scene = msg.data

    # Helper functions
    def get_time(self):
        self.central_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9

    def set_snapshot(self,snapshot):
        if not np.isscalar(snapshot):
            self.snapshot = torch.unsqueeze(torch.from_numpy(np.array(snapshot)).permute(2, 0, 1).float().to(self.device),0)
            
    
    def set_eye_pos(self,eye_pos):
        self.eye_pos = torch.unsqueeze(torch.from_numpy(np.array(eye_pos)).float().to(self.device),0)

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
                self.node_time = self.central_time   
                self.finished_pub.publish(Int32(data=self.node_id))
                continue

            # Run the classification model on the current snapshot and eye position
            log_softmax, self.recurrent = self.classmodel(self.snapshot, self.eye_pos, self.recurrent)
            log_softmax = log_softmax.detach().numpy()[0]
            class_probability = np.exp(log_softmax) 
        
            # Update the classification results
            self.update_classification_results(class_probability)
            
            # Update the node time and publish that the classification is finished
            self.node_time = self.central_time
            self.finished_pub.publish(Int32(data=self.node_id))

if __name__ == '__main__':
    rclpy.init()
    classifier_node = ClassifierROS2Node()
