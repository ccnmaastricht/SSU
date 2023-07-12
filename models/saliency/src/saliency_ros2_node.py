import json
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Float32

import numpy as np
from saliency_code import Saliency

class SaliencyROS2Node(Node):
    def __init__(self):
        super().__init__('saliency_node')
        
        # Set the 'use_sim_time' parameter to True
        use_sim_time = Parameter('use_sim_time', value=True)
        self.set_parameters([use_sim_time])

        # Initialize variables
        self.node_time = 0.0
        self.central_time = 0.0
        self.shut_down = False
        sef.waiting = False
        self.message = None


        with open("./saliency_model/parameters.json") as file:
            parameters = json.load(file)
        self.salmodel = Saliency(parameters)

        # publishers
        self.finished_pub = self.create_publisher(Bool, '/finished', 10)
        self.saliency_pub = self.create_publisher(Float32, '/saliency', 10)

        # subscribers
        self.snapshot_sub = self.create_subscription(Float32, '/camera_node/snapshot', self.snapshot_callback, 10)
        self.waiting_sub = self.create_subscription(Bool, '/camera_node/waiting', self.waiting_callback, 10)
        self.eye_pos_sub = self.create_subscription(Float32, '/saccade_node/eye_pos', self.eye_pos_callback, 10)
        self.shut_down_sub = self.create_subscription(Bool, '/sync_node/shutdown', self.shutdown_callback, 10)

        # start the loop
        self.saliency_loop()

    # Callback functions
    def snapshot_callback(self, msg):
        self.salmodel.set_input_tensor(msg.data)

    def waiting_callback(self, msg):
        self.waiting = msg.data
        
    def eye_pos_callback(self, msg):
        self.salmodel.set_eye_pos(msg.data)

    def shutdown_callback(self, msg):
        self.shut_down = msg.data

    # Helper functions
    def get_time(self):
        self.central_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9

    # Main loop
    def saliency_loop(self):
        while rclpy.ok():
            if self.shut_down:
                # Shutdown the node
                rclpy.shutdown()
                break

            rclpy.spin_once(self)

            self.get_time()
            
            if self.node_time>=self.central_time:
                # Wait for the next time step
                continue

            if self.waiting:
                # Wait for snapshot
                continue

            # Compute and publish the saliency map
            sal_map = self.salmodel.get_saliency_map()
            self.saliency_pub.publish(Float32(data=sal_map))
            
            # Update the node time and publish that the node has finished
            self.node_time = self.central_time        
            self.finished_pub.publish(Bool(data=True))

if __name__ == '__main__':
    rclpy.init()
    saliency_node = SaliencyROS2Node()