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

        self.salmodel = Saliency()