import cv2
import numpy as np
import tensorflow as tf

class Saliency:
    def __init__(self, parameters):
        '''
        Initialize the model
        '''
        self.model = tf.saved_model.load("./saliency_model")
        self.model = self.model.signatures["serving_default"]
        self.input_tensor = tf.ones((1, 240, 320, 3))
        self.eye_pos = np.array([0.0, 0.0])
        self.scene_size = (2048, 4096)
        self.snapshot_size = (1024, 1024)
        self.degrees_per_pixel = 0.087890625
        self.time_step = parameters["time_step"]
        self.decay_rate = parameters["decay_rate"]
        self.saliency_map = np.zeros(self.scene_size)

    def set_input_tensor(self, image):
        '''
        Set the input tensor to the model.

        Args:
            image (np.ndarray): The image to be used as input to the model
        '''

        image = cv2.resize(image, (320, 320))[40:280, :, :].astype(np.float32)
        self.input_tensor = tf.convert_to_tensor(np.expand_dims(image, axis=0))

    def set_eye_pos(self, eye_pos):
        '''
        Set the eye position of the model.

        Args:
            eye_pos (np.ndarray): The eye position
        '''

        self.eye_pos = eye_pos
        
    def get_saliency_map(self):
        '''
        Get the saliency map from the model.

        Returns:
            np.ndarray: The saliency map
        '''

        self.compute_global_saliency()
        return self.saliency_map
    
    def compute_local_saliency(self):
        '''
        Compute the local saliency of the snapshot.

        Returns:
            np.ndarray: Local saliency map
        '''

        return self.model(self.input_tensor)["output"].numpy().squeeze()
    
    def compute_global_saliency(self):
        '''
        Compute the global saliency of the snapshot.

        Returns:
            np.ndarray: The saliency map
        '''

        x = int(self.eye_pos[0] / self.degrees_per_pixel) + self.scene_size[1] // 2
        y = int(self.eye_pos[1] / self.degrees_per_pixel) + self.scene_size[0] // 2
        half_width = self.snapshot_size[0] // 2
        half_height = self.snapshot_size[1] // 2

        x_min = np.clip(x - half_width, 0, self.scene_size[1] - self.snapshot_size[0])
        x_max = x_min + self.snapshot_size[0]

        y_min = np.clip(y - half_height, 0, self.scene_size[0] - self.snapshot_size[1])
        y_max = y_min + self.snapshot_size[1]

        self.saliency_map *= np.exp(-self.time_step * self.decay_rate)
        local_saliency = self.compute_local_saliency()[:, 40:280]
        local_saliency = cv2.resize(local_saliency, self.snapshot_size, interpolation=cv2.INTER_CUBIC)
        self.saliency_map[y_min:y_max, x_min:x_max] = local_saliency
    