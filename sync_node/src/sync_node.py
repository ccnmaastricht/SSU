import json
import rclpy
import threading
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float32, Bool
from rosgraph_msgs.msg import Clock

class SyncROSNode(Node):
    def __init__(self):
        super().__init__('sync_node')

        self.finished_count = 0
        self.num_nodes = None
        self.time_step = None
        self.t_end = None
        self.current_time = None

        use_sim_time = Parameter('use_sim_time', value=True)
        self.set_parameters([use_sim_time])

        # publishers
        self.is_sim_pub = self.create_publisher(Bool, 'sync_node/issim', 10)
        self.shut_down_pub = self.create_publisher(Bool, '/sync_node/shutdown', 10)
        self.time_pub = self.create_publisher(Clock, '/clock', 10)
        
        # subscribers
        self.finished_sub = self.create_subscription(Bool, '/finished', self.finished_callback, 10)

        # Load config file
        self.load_config('sim_config.json')
        
 
    def publish_time(self):
        seconds = int(self.current_time)
        nanoseconds = int((self.current_time - seconds) * 1e9)
        msg = Clock()
        msg.clock = rclpy.time.Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
        self.time_pub.publish(msg)

    def finished_callback(self, msg):
        self.finished_count += 1
        if self.finished_count == self.num_nodes:
            self.advance_time()
            self.finished_count = 0

    def advance_time(self): ## make sure that sarting and finishing works properly across nodes
        self.current_time += self.time_step
   
    def shutdown(self):
        self.shut_down_pub.publish(Bool(data=True))
        rclpy.shutdown()

    def load_config(self, config_file):
        with open(config_file) as f:
            config = json.load(f)
        self.is_sim = config['is_sim']
        self.num_nodes = config['num_nodes']
        self.time_step = config['time_step']
        self.t_end = config['t_end']
        self.current_time = self.time_step

    def run_simulation(self):
        while self.current_time<self.t_end:
            self.publish_time()

        self.shutdown()


if __name__=='__main__':
    rclpy.init()    
    sync_node = SyncROSNode()

    simulation_thread = threading.Thread(target=sync_node.run_simulation)
    simulation_thread.start()

    rclpy.spin(sync_node)