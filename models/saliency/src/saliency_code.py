import numpy as np
import tensorflow as tf

class Saliency:
    def __init__(self):
        '''
        Initialize the model
        '''
        self.model = tf.saved_model.load("./saliency_model")
        self.model = self.model.signatures["serving_default"]
        self.input_tensor = tf.ones((1, 240, 320, 3))

    def set_input_tensor(self, image):
        '''
        Set the input tensor to the model.

        Args:
            image (np.ndarray): The image to be used as input to the model
        '''

        self.input_tensor = tf.convert_to_tensor(np.expand_dims(image, axis=0))
        
    def get_saliency(self):
        '''
        Get the saliency map from the model.

        Returns:
            np.ndarray: The saliency map
        '''
        
        return self.model(self.input_tensor)["output"].numpy().squeeze()