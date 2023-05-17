import cv2
import torch
import torch.nn as nn
import numpy as np

class SceneClassificationModel(nn.Module):
    def __init__(self, parameters):
        super(SceneClassificationModel, self).__init__()

        self.num_recurrent = parameters['num_recurrent']

        self.conv_stack1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=parameters['conv_stack1_out_channels'], kernel_size=parameters['conv_stack1_kernel_size'], padding=parameters['conv_stack1_padding']),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv_stack2 = nn.Sequential(
            nn.Conv2d(in_channels=parameters['conv_stack1_out_channels'], out_channels=parameters['conv_stack2_out_channels'], kernel_size=parameters['conv_stack2_kernel_size'], padding=parameters['conv_stack2_padding']),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.flatten = nn.Flatten()
        self.state_transition = (torch.randn(self.num_recurrent, self.num_recurrent) * 0.01).requires_grad_(True).cuda()
        #self.state_transition = nn.Linear(self.num_recurrent, self.num_recurrent)
        self.control_to_recurrent = nn.Linear(parameters['num_input'] + 2, self.num_recurrent)
        self.input_to_logit = nn.Linear(parameters['num_input'] + self.num_recurrent + 2, parameters['num_classes'])

    def forward(self, image, eye_pos, recurrent, time_step):
        '''Forward pass of the model
        Args:
            image (torch.Tensor): The input image
            eye_pos (torch.Tensor): The eye position
            recurrent (torch.Tensor): The recurrent state at the previous time step
            time_step (float): The time step to propagate the recurrent state by
        Returns:
            torch.Tensor: The class probabilities
            torch.Tensor: The recurrent state at the current time step
        '''

        input = self.conv_stack1(image)
        input = self.conv_stack2(input)
        input = self.flatten(input)
        control = torch.cat((input, eye_pos), dim=1)
        combined = torch.cat((control, recurrent), dim=1)

        recurrent = self.propagator(recurrent, time_step) + self.control_to_recurrent(control)

        logit = self.input_to_logit(combined)
        return logit, recurrent
    
    def init_recurrent(self):
        '''Initializes the recurrent state
        Args:
            batch_size (int): The batch size
        Returns:
            torch.Tensor: The initialized recurrent state
        '''

        return torch.zeros(1, self.num_recurrent).cuda()
    
    def propagator(self, recurrent, time_step):
        '''Propagates the recurrent state forward in time by a variable time step
        Args:
            recurrent (torch.Tensor): The recurrent state at the previous time step
            time_step (float): The time step to propagate the recurrent state by
        Returns:
            torch.Tensor: The recurrent state at the current time step
        '''

        recurrent = torch.matmul(recurrent, torch.linalg.matrix_exp(self.state_transition * time_step))
        return recurrent
    
def get_conv_output_size(input_size, kernel_size, padding, stride):
    '''Calculates the output size of a convolutional layer
    Args:
        input_size (int): The input size
        kernel_size (int): The kernel size
        padding (int): The padding
        stride (int): The stride
    Returns:
        int: The output size
    '''
    return int((input_size - kernel_size + 2 * padding) / stride) + 1

def softmax(x):
    '''Computes the softmax function
    Args:
        x (np.ndarray): The input
    Returns:
        np.ndarray: The output
    '''
    return np.exp(x) / np.sum(np.exp(x), axis=1)

def load_data(data_dir):
    '''Loads training and validation data
    Args:
        data_dir (str): The data directory
    Returns:
        tuple: The training data and labels
        tuple: The validation data and labels
        tuple: Number of training and validation sequences
    '''

    train_dir = f'{data_dir}/train'
    val_dir = f'{data_dir}/val'

    train_labels = np.load(f'{train_dir}/labels.npy')
    num_train_sequences = train_labels.shape[0]

    val_labels = np.load(f'{val_dir}/labels.npy')
    num_val_sequences = val_labels.shape[0]

    train_data = []
    val_data = []

    for i in range(num_train_sequences):
        durations = np.load(f'{train_dir}/sequence_{i}/durations.npy')
        num_items = durations.shape[0]
        coordinates = np.load(f'{train_dir}/sequence_{i}/coordinates.npz')
        eye_pos = np.zeros((num_items, 2))
        eye_pos[:, 0] = coordinates['x_coord']
        eye_pos[:, 1] = coordinates['y_coord']
        dict = {'snapshots': [], 'eye_pos': eye_pos, 'time_steps': durations}
        for j in range(num_items):
            img = cv2.imread(f'{train_dir}/sequence_{i}/snapshot_{j}.png')
            dict['snapshots'].append(img)

        train_data.append(dict)

    for i in range(num_val_sequences):
        durations = np.load(f'{val_dir}/sequence_{i}/durations.npy')
        num_items = durations.shape[0]
        coordinates = np.load(f'{val_dir}/sequence_{i}/coordinates.npz')
        eye_pos = np.zeros((num_items, 2))
        eye_pos[:, 0] = coordinates['x_coord']
        eye_pos[:, 1] = coordinates['y_coord']
        dict = {'snapshots': [], 'eye_pos': eye_pos, 'time_steps': durations}
        for j in range(num_items):
            img = cv2.imread(f'{val_dir}/sequence_{i}/snapshot_{j}.png')
            dict['snapshots'].append(img)

        val_data.append(dict)

    return (train_data, train_labels), (val_data, val_labels), (num_train_sequences, num_val_sequences)

def shuffler(data, labels, num_sequences):
    '''Shuffles the data
    Args:
        data (list): The data
        labels (np.ndarray): The labels
        num_sequences (int): The number of sequences
    Returns:
        list: The shuffled data
        np.ndarray: The shuffled labels
    '''

    indices = np.arange(num_sequences)
    np.random.shuffle(indices)
    shuffled_data = []
    shuffled_labels = np.zeros_like(labels)
    for i in range(num_sequences):
        shuffled_data.append(data[indices[i]])
        shuffled_labels[i] = labels[indices[i]]

    return shuffled_data, shuffled_labels

def sample_to_tensor(sample):
    '''Converts a sample to a tensor
    Args:
        sample (dict): The sample
    Returns:
        torch.Tensor: The tensor
    '''

    snapshots = torch.from_numpy(np.array(sample['snapshots'])).permute(0, 3, 1, 2).float().cuda()
    eye_positions = torch.from_numpy(sample['eye_pos']).float().cuda()
    time_steps = torch.from_numpy(sample['time_steps']).float().cuda()
    return snapshots, eye_positions, time_steps

def main():

    imfile = '/media/mario/HDD/Data/360Panorama_2D-3D-S/sequences/train/sequence_0/snapshot_7.png'
    img = cv2.imread(imfile)

    conv_stack1_out_channels = 16
    conv_stack1_kernel_size = 7
    conv_stack1_padding = 3
    conv_stack1_stride = 1
    conv_stack2_out_channels = 32
    conv_stack2_kernel_size = 7
    conv_stack2_padding = 2
    conv_stack2_stride = 1
    max_pool_kernel_size = 2
    max_pool_stride = 2

    image_size = 256

    conv_stack1_output_size = get_conv_output_size(image_size, conv_stack1_kernel_size, conv_stack1_padding, conv_stack1_stride)
    max_pool1_output_size = get_conv_output_size(conv_stack1_output_size, max_pool_kernel_size, 0, max_pool_stride)
    conv_stack2_output_size = get_conv_output_size(max_pool1_output_size, conv_stack2_kernel_size, conv_stack2_padding, conv_stack2_stride)
    max_pool2_output_size = get_conv_output_size(conv_stack2_output_size, max_pool_kernel_size, 0, max_pool_stride)
    num_input = max_pool2_output_size * max_pool2_output_size * conv_stack2_out_channels

    parameters = {
        'conv_stack1_out_channels': conv_stack1_out_channels,
        'conv_stack1_kernel_size': conv_stack1_kernel_size,
        'conv_stack1_padding': conv_stack1_padding,
        'conv_stack2_out_channels': conv_stack2_out_channels,
        'conv_stack2_kernel_size': conv_stack2_kernel_size,
        'conv_stack2_padding': conv_stack2_padding,
        'num_input': num_input,
        'num_recurrent': 32,
        'num_classes': 11
    }
    model = SceneClassificationModel(parameters).cuda()
    recurrent = model.init_recurrent()


    train, val, num_sequences = load_data('/media/mario/HDD/Data/360Panorama_2D-3D-S/sequences')
    train_data, train_labels = train
    num_train_sequences, _ = num_sequences

    shuffled_train_data, shuffled_train_labels = shuffler(train_data, train_labels, num_train_sequences)
    snapshots, eye_positions, time_steps = sample_to_tensor(shuffled_train_data[0])

    logit, recurrent = model(snapshots[0], torch.unsqueeze(eye_positions[0],0), recurrent, time_steps[0])
    print(logit)

   


if __name__ == '__main__':
    main()


'''
SOMETHING IS OFF WITH DATA PREPARATION

python train_model.py
Traceback (most recent call last):
  File "train_model.py", line 235, in <module>
    main()
  File "train_model.py", line 228, in main
    logit, recurrent = model(snapshots[0], torch.unsqueeze(eye_positions[0],0), recurrent, time_steps[0])
  File "/home/mario/miniconda3/envs/torch_env/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "train_model.py", line 45, in forward
    control = torch.cat((input, eye_pos), dim=1)
RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 32 but got size 1 for tensor number 1 in the list.
'''

#\q: what is the problem?
#\a: the problem is that the eye_pos is a 1D tensor, but the input is a 4D tensor. The eye_pos needs to be expanded to 4D.