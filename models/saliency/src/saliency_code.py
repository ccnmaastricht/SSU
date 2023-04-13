import numpy as np
import tensorflow as tf

class Saliency:
    def __init__(self):
        self.model = tf.saved_model.load("./saliency_model")
        self.model = self.model.signatures["serving_default"]
        self.input_tensor = tf.ones((1, 240, 320, 3))

    def set_input_tensor(self, image):
        self.input_tensor = tf.convert_to_tensor(np.expand_dims(image, axis=0))
        
    def get_saliency(self):
        return self.model(self.input_tensor)["output"].numpy().squeeze()