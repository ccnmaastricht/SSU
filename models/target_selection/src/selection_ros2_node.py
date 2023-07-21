import json
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Int32, Float32MultiArray

import numpy as np
from selection_code import TargetSelection

class SelectionROS2Node(Node):
    def __init__(self):
        super().__init__('selection_node')
        
        # Set the 'use_sim_time' parameter to True
        use_sim_time = Parameter('use_sim_time', value=True)
        self.set_parameters([use_sim_time])

        # Initialize variables
        self.node_time = 0.0
        self.central_time = 0.0
        self.shut_down = False
        self.waiting = True
        self.saliency = None
        self.node_id = 5

        self.model = TargetSelection()

        # publishers
        self.finished_pub = self.create_publisher(Int32, '/finished', 10)
        self.target_pub = self.create_publisher(Float32MultiArray, '/target_location', 10)

        # subscribers
        self.snapshot_sub = self.create_subscription(Float32MultiArray, '/saliency_node/saliency', self.saliency_callback, 10)
        self.shut_down_sub = self.create_subscription(Bool, '/sync_node/shutdown', self.shutdown_callback, 10)

        # start the loop
        self.selection_loop()

    # Callback functions
    def saliency_callback(self, msg):
        self.saliency = np.array(msg.data).reshape(2048, 4096)
        
    def shutdown_callback(self, msg):
        self.shut_down = msg.data

    # Helper functions
    def get_time(self):
        self.central_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9

    # Main loop
    def selection_loop(self):
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

            if (self.waiting) or (self.saliency is None):
                # Wait for snapshot
                self.node_time = self.central_time   
                self.finished_pub.publish(Int32(data=self.node_id))
                continue

            # Compute and publish eye position
            target_location = self.model.sample_location(self.saliency)
            self.target_pub.publish(Float32MultiArray(data=target_location))
            
            # Update the node time and publish that the node has finished
            self.node_time = self.central_time        
            self.finished_pub.publish(Int32(data=self.node_id))

if __name__ == '__main__':
    rclpy.init()
    selection_node = SelectionROS2Node()