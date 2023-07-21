import json
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Int32, Float32MultiArray, MultiArrayDimension

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
        self.waiting = True
        self.node_id = 3

        with open("./saliency_model/parameters.json") as file:
            parameters = json.load(file)
        self.salmodel = Saliency(parameters)

        # publishers
        self.finished_pub = self.create_publisher(Int32, '/finished', 10)
        self.saliency_pub = self.create_publisher(Float32MultiArray, '/saliency', 10)

        # subscribers
        self.snapshot_sub = self.create_subscription(Float32MultiArray, '/camera_node/snapshot', self.snapshot_callback, 10)
        self.waiting_sub = self.create_subscription(Bool, '/camera_node/waiting', self.waiting_callback, 10)
        self.eye_pos_sub = self.create_subscription(Float32MultiArray, '/saccade_node/eye_pos', self.eye_pos_callback, 10)
        self.shut_down_sub = self.create_subscription(Bool, '/sync_node/shutdown', self.shutdown_callback, 10)

        # start the loop
        self.saliency_loop()

    def publish_salience(self, sal_map):
        msg = Float32MultiArray()
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].label = "rows"
        msg.layout.dim[0].size = sal_map.shape[0]
        msg.layout.dim[0].stride = sal_map.shape[0] * sal_map.shape[1]
        msg.layout.dim[1].label = "cols"
        msg.layout.dim[1].size = sal_map.shape[1]
        msg.layout.dim[1].stride = sal_map.shape[1]
        msg.data = sal_map.reshape(sal_map.size).tolist()
        self.saliency_pub.publish(msg)


    # Callback functions
    def snapshot_callback(self, msg):
        # Convert the data field to a NumPy array and reshape it to its original shape
        snapshot = np.array(msg.data)
        snapshot = snapshot.reshape((msg.layout.dim[0].size, msg.layout.dim[1].size, msg.layout.dim[2].size))
        
        # Set the input tensor of the saliency model
        self.salmodel.set_input_tensor(snapshot)

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

            print('saliency node - waiting for snapshot')
            rclpy.spin_once(self)

            self.get_time()
            print(f'saliency node - node time: {self.node_time}, central time: {self.central_time}')
            
            if self.node_time>=self.central_time:
                # Wait for the next time step
                continue

            if self.waiting:
                # Wait for snapshot
                self.finished_pub.publish(Bool(data=True))
                continue

            # Compute and publish the saliency map
            sal_map = self.salmodel.get_saliency_map()
            self.publish_salience(sal_map)
            
            # Update the node time and publish that the node has finished
            self.node_time = self.central_time        
            self.finished_pub.publish(Int32(data=self.node_id))

if __name__ == '__main__':
    rclpy.init()
    saliency_node = SaliencyROS2Node()