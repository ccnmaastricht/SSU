import numpy as np

class TargetSelection():
    def __init__(self):
        '''
        Initialize the target selection model.
        '''
        
        self.degrees_per_pixel = 0.087890625
        self.pixels = 4096

    
    def sample_location(self, saliency):
        '''
        Sample fixation location based on saliency map.

        Args:
            saliency (np.ndarray): The saliency map

        Returns:
            list: The sampled location in degrees
        '''
        
        sample = (np.random.random(saliency.shape) * saliency).flatten()
        index = np.argmax(sample)
        
        return [index // self.pixels * self.degrees_per_pixel, index % self.pixels * self.degrees_per_pixel]
    
    
