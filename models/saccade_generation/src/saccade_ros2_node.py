import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Int32, Float32MultiArray

import numpy as np
from saccade_code import SaccadeGenerator

class SaccadeROS2Node(Node):
    def __init__(self):
        '''
        Initialize the SaccadeROS2Node.

        Creates publishers and subscribers, initializes variables, and starts the main loop.
        '''
        super().__init__('saccade_node')
        
        # Set the 'use_sim_time' parameter to True
        use_sim_time = Parameter('use_sim_time', value=True)
        self.set_parameters([use_sim_time])

        # Initialize variables
        self.node_time = 0.0
        self.central_time = 0.0
        self.shut_down = False
        self.node_id = 2

        self.saccade_generator = SaccadeGenerator()
        self.target_location = [0, 0]

        self.horizontal_factor = 5.0
        self.vertical_factor = 5.0
        self.min_current = 300.0

        # publishers
        self.finished_pub = self.create_publisher(Int32, '/finished', 10)
        self.eye_pos_pub = self.create_publisher(Float32MultiArray, 'saccade_node/eye_pos', 10)

        # subscribers
        self.target_location_sub = self.create_subscription(Float32MultiArray, '/selection_node/target_location', self.target_location_callback, 10)
        self.shut_down_sub = self.create_subscription(Bool, '/sync_node/shutdown', self.shutdown_callback, 10)

        # start the loop
        self.saccade_loop()

    # Callback functions
    def target_location_callback(self, msg):
        '''
        Callback function for the '/selection_node/target_location' topic.

        Args:
            msg (Float32MultiArray): The target location
        '''
        self.target_location = msg.data

    def shutdown_callback(self, msg):
        '''
        Callback function for the '/sync_node/shutdown' topic.

        Args:
            msg (Bool): The shutdown message
        '''
        self.shut_down = msg.data

    # Helper functions
    def get_time(self):
        '''
        Get the current time in seconds.
        '''
        self.central_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9

    def compute_displacement(self):
        '''
        Compute the desired displacement of the eye.

        Returns:
            horizontal_displacement (float): The desired horizontal displacement of the eye
            vertical_displacement (float): The desired vertical displacement of the eye
        '''
        horizontal = self.saccade_generator.eye_position[0]
        vertical = self.saccade_generator.eye_position[1]
        horizontal_displacement = self.target_location[0] - horizontal
        vertical_displacement = self.target_location[1] - vertical

        return horizontal_displacement, vertical_displacement
    
    def compute_input_current(self, horizontal_displacement, vertical_displacement):
        '''
        Compute the input current to the saccade generator.

        Args:
            horizontal_displacement (float): The desired horizontal displacement of the eye
            vertical_displacement (float): The desired vertical displacement of the eye

        Returns:
            current_left (float): The input current to the left muscle
            current_right (float): The input current to the right muscle
            current_up (float): The input current to the up muscle
            current_down (float): The input current to the down muscle
        '''
        current_left = -np.minimum(horizontal_displacement, 0.) * self.horizontal_factor + self.min_current
        current_right = np.maximum(horizontal_displacement, 0.) * self.horizontal_factor + self.min_current
        current_up = np.maximum(vertical_displacement, 0.) * self.vertical_factor + self.min_current
        current_down = -np.minimum(vertical_displacement, 0.) * self.vertical_factor + self.min_current

        return (current_left, current_right, current_up, current_down)


    # Main loop
    def saccade_loop(self):
        '''
        The main loop of the SaccadeROS2Node.

        It waits for the node to receive a target location from the SelectionROS2Node, then computes the desired displacement and input current. 
        It then simulates the saccade generator and publishes the eye position.
        '''


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
            
            self.eye_pos_pub.publish(Float32MultiArray(data=self.saccade_generator.eye_position.tolist()))
            
            # Update the node time and publish that the node has finished
            self.node_time = self.central_time
            self.finished_pub.publish(Int32(data=self.node_id))
            
                  

if __name__ == '__main__':
    rclpy.init()
    saccade_node = SaccadeROS2Node()