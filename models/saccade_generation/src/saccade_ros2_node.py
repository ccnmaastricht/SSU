import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Float32

import numpy as np
from saccade_generator import SaccadeGenerator

class SaccadeROS2Node(Node):
    def __init__(self):
        super().__init__('saccade_node')
        
        # Set the 'use_sim_time' parameter to True
        use_sim_time = Parameter('use_sim_time', value=True)
        self.set_parameters([use_sim_time])

        # Initialize variables
        self.node_time = 0.0
        self.central_time = 0.0
        self.shut_down = False
        self.message = None

        self.saccade_generator = SaccadeGenerator()
        self.target_location = np.zeros(2)

        self.horizontal_factor = 5.0
        self.vertical_factor = 2.5
        self.min_current = 300.0

        # publishers
        self.finished_pub = self.create_publisher(Bool, '/finished', 10)
        self.eye_pos_pub = self.create_publisher(Float32, '/eye_pos', 10)

        # subscribers
        self.target_location_sub = self.create_subscription(Float32, '/selection_node/target_location', self.target_location_callback, 10)
        self.shut_down_sub = self.create_subscription(Bool, '/sync_node/shutdown', self.shutdown_callback, 10)

        # start the loop
        self.saliency_loop()

    # Callback functions
    def target_location_callback(self, msg):
        self.target_location = msg.data

    def shutdown_callback(self, msg):
        self.shut_down = msg.data

    # Helper functions
    def get_time(self):
        self.central_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9

    def compute_displacement(self):
        horizontal = self.saccade_generator.eye_position[0]
        vertical = self.saccade_generator.eye_position[1]
        horizontal_displacement = self.target_location[0] - horizontal
        vertical_displacement = self.target_location[1] - vertical

        return horizontal_displacement, vertical_displacement
    
    def compute_input_current(self, horizontal_displacement, vertical_displacement):
        current_left = -np.minimum(horizontal_displacement, 0.) * self.horizontal_factor + self.min_current
        amp_right = np.maximum(horizontal_displacement, 0.) * self.horizontal_factor + self.min_current
        amp_up = np.maximum(vertical_displacement, 0.) * self.vertical_factor + self.min_current
        amp_down = -np.minimum(vertical_displacement, 0.) * self.vertical_factor + self.min_current

        return (current_left, amp_right, amp_up, amp_down)


    # Main loop
    def saccade_loop(self):
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

            # Compute and publish the eye position
            desired_displacement = self.compute_displacement()
            input_current = self.compute_input_current(*desired_displacement)
            self.saccade_generator.simulate(input_current)
            self.eye_pos_pub.publish(Float32(data=self.saccade_generator.eye_position))
            
            # Update the node time and publish that the node has finished
            self.node_time = self.central_time        
            self.finished_pub.publish(Bool(data=True))

if __name__ == '__main__':
    rclpy.init()
    saliency_node = SaccadeROS2Node()