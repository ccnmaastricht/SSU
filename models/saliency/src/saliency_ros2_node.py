import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Float32
from saliency_code import Saliency

class SaliencyROS2Node(Node):
    def __init__(self):
        super().__init__('saliency_node')
        
        use_sim_time = Parameter('use_sim_time', value=True)
        self.set_parameters([use_sim_time])

        self.node_time = 0.0
        self.central_time = 0.0
        self.shut_down = False
        self.message = None
        self.salmodel = Saliency()

        # publishers
        self.finished_pub = self.create_publisher(Bool, '/finished', 10)
        self.saliency_pub = self.create_publisher(Float32, '/saliency', 10)

        # subscribers
        self.snapshot_sub = self.create_subscription(Float32, '/camera_node/snapshot', self.snapshot_callback, 10)
        self.shut_down_sub = self.create_subscription(Bool, '/sync_node/shutdown', self.shutdown_callback, 10)

        # start the loop
        self.saliency_loop()

    # functions
    def snapshot_callback(self, msg):
        self.salmodel.set_input_tensor(msg.data)

    def shutdown_callback(self, msg):
        self.shut_down = msg.data

    def get_time(self):
        self.central_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9

    def saliency_loop(self):
        while rclpy.ok():
            if self.shut_down:
                rclpy.shutdown()
                break

            rclpy.spin_once(self)

            self.get_time()
            
            if self.node_time>=self.central_time:
                continue

            sal_map = self.salmodel.get_saliency()
            self.saliency_pub.publish(Float32(data=sal_map))
            
            self.node_time = self.central_time
            
            self.finished_pub.publish(Bool(data=True))

if __name__ == '__main__':
    rclpy.init()
    saliency_node = SaliencyROS2Node()