import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, String, Int32, Float32MultiArray,  MultiArrayDimension
from camera_code import Camera

class CameraROS2Node(Node):
    def __init__(self):
        super().__init__('camera_node')
        
        # Set the 'use_sim_time' parameter to True
        use_sim_time = Parameter('use_sim_time', value=True)
        self.set_parameters([use_sim_time])

        # Initialize variables
        self.node_time = 0.0
        self.central_time = 0.0
        self.shut_down = False
        self.node_id = 1

        self.camera = Camera()
        
        # publishers
        self.snapshot_pub = self.create_publisher(Float32MultiArray, '/camera_node/snapshot', 10)
        self.waiting_pub = self.create_publisher(Bool, '/camera_node/waiting', 10)
        self.finished_pub = self.create_publisher(Int32, '/finished', 10)

        # subscribers
        self.shut_down_sub = self.create_subscription(Bool, '/sync_node/shutdown', self.shutdown_callback, 10)
        self.scene_sub = self.create_subscription(String, '/sync_node/scene', self.scene_callback, 10)
        self.target_location_sub = self.create_subscription(Float32MultiArray, '/selection_node/target_location', self.target_location_callback, 10)
        self.eye_pos_sub = self.create_subscription(Float32MultiArray, '/saccade_node/eye_pos', self.eye_pos_callback, 10)

        # start the loop
        self.camera_loop()

    def publish_snapshot(self, snapshot):
        msg = Float32MultiArray()
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].label = "rows"
        msg.layout.dim[0].size = snapshot.shape[0]
        msg.layout.dim[0].stride = snapshot.shape[0] * snapshot.shape[1] * snapshot.shape[2]
        msg.layout.dim[1].label = "cols"
        msg.layout.dim[1].size = snapshot.shape[1]
        msg.layout.dim[1].stride = snapshot.shape[1] * snapshot.shape[2]
        msg.layout.dim[2].label = "depth"
        msg.layout.dim[2].size = snapshot.shape[2]
        msg.layout.dim[2].stride = snapshot.shape[2]
        msg.data = snapshot.reshape(snapshot.size).tolist()
        self.snapshot_pub.publish(msg)


    # Callback functions
    def shutdown_callback(self, msg):
        self.shut_down = msg.data

    def scene_callback(self, msg):
        file = os.path.join("/usr/data", msg.data, "image.png")
        self.camera.set_scene(file)

    def target_location_callback(self, msg):
        self.camera.set_target_location(msg.data)

    def eye_pos_callback(self, msg):
        self.camera.set_eye_pos(msg.data)


    # Helper functions
    def get_time(self):
        self.central_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9

    # Main loop
    def camera_loop(self):
        snapshot = np.zeros((1024, 1024, 3))
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

            # Compute distance to target and publish waiting
            distance = self.camera.compute_distance()
            if (distance > 1.0) or (self.camera.scene is None):
                self.waiting_pub.publish(Bool(data=True))
                self.node_time = self.central_time   
                self.finished_pub.publish(Int32(data=self.node_id))
                continue
            
            self.waiting_pub.publish(Bool(data=False))
            
            # Extract the current snapshot and publish it
            snapshot = self.camera.get_snapshot()
            self.publish_snapshot(snapshot)
            
            # Update the node time and publish that the node has finished
            self.node_time = self.central_time        
            self.finished_pub.publish(Int32(data=self.node_id))

if __name__ == '__main__':
    rclpy.init()
    camera_node = CameraROS2Node()
