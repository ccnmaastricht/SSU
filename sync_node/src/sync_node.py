import os
import json
import rclpy
import threading
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, String, Int32
from rosgraph_msgs.msg import Clock

class SyncROSNode(Node):
    def __init__(self):
        '''
        Initialize the SyncROSNode.
        '''
        super().__init__('sync_node')

        self.finished_count = 0
        self.finished_nodes = set()
        self.current_time = 0.0

        use_sim_time = Parameter('use_sim_time', value=True)
        self.set_parameters([use_sim_time])

        # publishers
        self.scene_pub = self.create_publisher(String, 'sync_node/scene', 10)
        self.shut_down_pub = self.create_publisher(Bool, '/sync_node/shutdown', 10)
        self.time_pub = self.create_publisher(Clock, '/clock', 10)
        
        # subscribers
        self.finished_sub = self.create_subscription(Int32, '/finished', self.finished_callback, 10)

        # Load config file
        file = os.path.join("/usr/config", "simulation_configuration.json")
        self.load_config(file)
        
 
    def publish_time(self):
        '''
        Publish the current time to the '/clock' topic.
        '''
        seconds = int(self.current_time)
        nanoseconds = int((self.current_time - seconds) * 1e9)
        msg = Clock()
        msg.clock = rclpy.time.Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
        self.time_pub.publish(msg)

    def finished_callback(self, msg):
        '''
        Callback function for the '/finished' topic.

        Args:
            msg (Int32): ID of node that finished.
        '''
        if msg.data not in self.finished_nodes:
            self.finished_nodes.add(msg.data)
            self.finished_count += 1
        if self.finished_count == self.num_nodes:
            self.advance_time()
            self.finished_nodes.clear()
            self.finished_count = 0
        

    def advance_time(self):
        '''
        Advance the current time by the time step.
        '''
        self.current_time += self.time_step
   
    def shutdown(self):
        '''
        Shutdown the simulation.
        '''
        self.shut_down_pub.publish(Bool(data=True))
        rclpy.shutdown()

    def load_config(self, config_file):
        '''
        Load the simulation configuration file.

        Args:
            config_file (str): Path to the configuration file.
        '''
        with open(config_file) as f:
            config = json.load(f)
        self.num_nodes = config['num_nodes']
        self.time_step = config['time_step']
        self.viewing_time = config['viewing_time']
        self.scene = config['scene']
        self.current_time = self.time_step

    def run_simulation(self):
        '''
        Run the simulation.
        '''
        while self.current_time<self.viewing_time:
            self.scene_pub.publish(String(data=self.scene))
            self.publish_time()
            
        self.shutdown()


if __name__=='__main__':
    rclpy.init()    
    sync_node = SyncROSNode()

    simulation_thread = threading.Thread(target=sync_node.run_simulation)
    simulation_thread.start()

    rclpy.spin(sync_node)