import numpy as np
from cv2 import dnn_superres
from scipy.sparse import lil_matrix as sparse


class GanglionSampling:
    '''
    Class that implements the retinal sampling model described in Watson (2014). The model is used to convert
    images from a given resolution to a new resolution. The model is based on the fact that the human retina
    has a limited amount of Ganglion cells. The model uses the Ganglion cell density function from Watson (2014)
    to determine the amount of cells involved in processing the image along the radius given the covered visual
    field. The model then uses this information to determine the new location of each pixel in the new image.
    '''

    def __init__(self, parameters):
        self.r2 = 1.05                  # The eccentricity at which density is reduced by a factor of four (and spacing is doubled)
        self.re = 22                    # Scale factor of the exponential. Not used in our version.
        self.dg = 33162                 # Cell density at r = 0
        self.C = (3.4820e+04 + .1)      # Constant added to the integral to make sure that if x == 0, y == 0.
        self.W = None                   # Placeholder for spare matrix used for image transformation when using series_dist method
        self.msk = None                 # Placeholder for mask when using series_dist method
        
        self.__generate_sampler(parameters)
        self.__generate_upscaler(upscaling_factor=parameters['upscaling_factor'])

    def fi(self, r):
        '''
        Integrated Ganglion Density formula from Watson (2014). Maps from degrees of visual angle to the amount of cells.

        Args:
            r (float): The radius of the image in degrees of visual angle.

        Returns:
            float: The amount of cells involved in processing the image along the radius given the covered visual field.
        '''
        return self.C - np.divide((np.multiply(self.dg, self.r2 ** 2)), (r + self.r2))

    def fii(self, r):
        '''
        Inverted integrated Ganglion Density formula from Watson (2014). Maps from the amount of cells to degrees of visual angle.

        Args:
            r (float): The amount of cells involved in processing the image along the radius given the covered visual field.

        Returns:
            float: The radius of the image in degrees of visual angle.
        '''

        return np.divide(np.multiply(self.dg, self.r2 ** 2), (self.C - r)) - self.r2

    def __generate_upscaler(self, upscaling_factor=None):
        '''
        Generates an upscaler object that can be used to upsample images.
        '''

        if upscaling_factor is None:
            self.upscaler = None
            return  # No upscaling is needed

        # Create an SR object
        self.upscaler = dnn_superres.DnnSuperResImpl_create()

        # Set the desired model and scale to get correct pre- and post-processing
        if upscaling_factor == 2:
            path = "upscaling/LapSRN_x2.pb"
            self.upscaler.readModel(path)
            self.upscaler.setModel("lapsrn", 2)
        elif upscaling_factor == 4:
            path = "upscaling/LapSRN_x4.pb"
            self.upscaler.readModel(path)
            self.upscaler.setModel("lapsrn", 4)
        elif upscaling_factor == 8:
            path = "upscaling/LapSRN_x8.pb"
            self.upscaler.readModel(path)
            self.upscaler.setModel("lapsrn", 8)
        else:
            raise ValueError("Upscaling factor must be either 2, 4 or 8")
        
        

    def __generate_sampler(self, parameters):
        '''
        Generates a sparse matrix that maps every pixel in the old image, to each pixel in the new image via inverse mapping.

        Args:
            parameters (dict): Dictionary containing the parameters for the model.
        '''

        # Unpack parameters
        self.in_res, self.depth = parameters['input_dims']  # Input image dimensions
        self.out_res = parameters['output_dims']            # Output image dimensions
        fov = parameters['fov']                             # Field of view in degrees of visual angle

        # Calculate various parameters for the sampler
        vf_radius = fov / 2                                 # Field of view radius in degrees of visual angle
        in_radius = self.in_res / 2                         # Input image radius in pixels
        n_cells = self.fi(vf_radius)                        # Number of cells along the radius of the covered field of view
        deg_per_pixel = vf_radius / in_radius               # Degrees of visual angle per pixel

        # Create a grid for the output image
        t = np.linspace(-n_cells, n_cells, num=self.out_res) 
        x, y = np.meshgrid(t, t)
        x = np.reshape(x, self.out_res ** 2)
        y = np.reshape(y, self.out_res ** 2)

        # Calculate the angle and radius of each pixel
        ang = np.angle(x + y * 1j)
        rad = np.abs(x + y * 1j)

        # Calculate new pixel locations using inverse mapping
        new_r = self.fii(rad) / deg_per_pixel
        x_n = np.multiply(np.cos(ang), new_r) + self.in_res / 2
        y_n = np.multiply(np.sin(ang), new_r) + self.in_res / 2
            

        # Build the sparse matrix for image conversion
        self.W = sparse((self.out_res ** 2, self.in_res ** 2), dtype=float)
        for i in range(self.out_res ** 2):
            x = np.minimum(np.maximum([np.floor(y_n[i]), np.ceil(y_n[i])], 0), self.in_res - 1)
            y = np.minimum(np.maximum([np.floor(x_n[i]), np.ceil(x_n[i])], 0), self.in_res - 1)
            c, idx = np.unique([x[0] * self.in_res + y, x[1] * self.in_res + y], return_index=True)
            dist = np.reshape(np.array([np.abs(x - x_n[i]), np.abs(y - y_n[i])]), 4)
            dist_sum = sum(dist[idx])
            self.W[i, c] = dist[idx] / dist_sum if dist_sum > 0 else 0

        
    def resample_image(self, image):
        '''
        Resamples the image to the new resolution.

        Args:
            image (array_like): Numpy array containing an image. Leaving it empty will call up a GUI to manually select a file.

        Returns:
            array_like: Resampled image.
        '''
   
        # Upscale the image
        if self.upscaler is not None:
            image = self.upscaler.upsample(image)

        # Vectorize the image
        image = np.reshape(image, (self.in_res ** 2, self.depth)).squeeze()
        
        # Resample the image
        return self.W.dot(image).reshape(self.out_res, self.out_res, self.depth).astype(np.uint8)

    