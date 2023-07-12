import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Float32
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
        self.message = None

        self.camera = Camera()
        self.camera.set_scene('placeholder')


        # publishers
        self.snapshot_pub = self.create_publisher(Float32, '/snapshot', 10)
        self.waiting_pub = self.create_publisher(Bool, '/waiting', 10)
        self.finished_pub = self.create_publisher(Bool, '/finished', 10)

        # subscribers
        self.shut_down_sub = self.create_subscription(Bool, '/sync_node/shutdown', self.shutdown_callback, 10)
        self.target_location_sub = self.create_subscription(Float32, '/selection_node/target_location', self.target_location_callback, 10)
        self.eye_pos_sub = self.create_subscription(Float32, '/saccade_node/eye_pos', self.eye_pos_callback, 10)

        # start the loop
        self.camera_loop()

    # Callback functions
    def shutdown_callback(self, msg):
        self.shut_down = msg.data

    def target_location_callback(self, msg):
        self.camera.set_target_location(msg.data)

    def eye_pos_callback(self, msg):
        self.camera.set_eye_pos(msg.data)


    # Helper functions
    def get_time(self):
        self.central_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9

    # Main loop
    def camera_loop(self):

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
            if distance > 1.0:
                self.waiting_pub.publish(Bool(data=True))
                continue
            else:
                self.waiting_pub.publish(Bool(data=False))
            
            # Extract and publish the current snapshot
            snapshot = self.camera.get_snapshot()
            self.snapshot_pub.publish(Float32(data=snapshot))

            
            # Update the node time and publish that the node has finished
            self.node_time = self.central_time        
            self.finished_pub.publish(Bool(data=True))

if __name__ == '__main__':
    rclpy.init()
    camera_node = CameraROS2Node()
