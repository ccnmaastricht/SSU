import cv2
import numpy as np

class Camera:
    def __init__(self):
        '''
        Initialize the camera
        '''
        self.snapshot_size = (1024, 1024)
        self.scene_size = (2048, 4096)
        self.degrees_per_pixel = 0.087890625
        self.eye_pos = np.array([0.0, 0.0])
        self.target_location = np.array([0.0, 0.0])
        self.scene = None

    def set_scene(self, scene_path):
        '''
        Set the scene to be explored by the camera.

        Args:
            scene_path (string): The path to the scene file
        '''
        scene = cv2.imread(scene_path)
        self.scene = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)

    def set_eye_pos(self, eye_pos):
        '''
        Set the eye position of the camera.

        Args:
            eye_pos (list): The eye position
        '''
        self.eye_pos = np.array(eye_pos)

    def set_target_location(self, target_location):
        '''
        Set the target location of the camera.

        Args:
            target_location (list): The target location
        '''
        self.target_location = np.array(target_location)

    def compute_distance(self):
        '''
        Compute the distance between the eye position and the target location.

        Returns:
            float: The distance
        '''
        return np.linalg.norm(self.eye_pos - self.target_location)
    

    def get_snapshot(self):
        '''
        Get a snapshot from the camera.

        Returns:
            np.ndarray: The snapshot
        '''

        x = int(self.eye_pos[0] / self.degrees_per_pixel) + self.scene_size[1] // 2
        y = int(self.eye_pos[1] / self.degrees_per_pixel) + self.scene_size[0] // 2
        half_width = self.snapshot_size[0] // 2
        half_height = self.snapshot_size[1] // 2

        x_min = np.clip(x - half_width, 0, self.scene_size[1] - self.snapshot_size[0])
        x_max = x_min + self.snapshot_size[0]

        y_min = np.clip(y - half_height, 0, self.scene_size[0] - self.snapshot_size[1])
        y_max = y_min + self.snapshot_size[1]

        snapshot = self.scene[y_min:y_max, x_min:x_max].astype('float32')
        return snapshot
