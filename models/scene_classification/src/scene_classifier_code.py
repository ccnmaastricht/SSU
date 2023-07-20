import torch
import torch.nn as nn


class SceneClassificationModel(nn.Module):
    def __init__(self, parameters):
        super(SceneClassificationModel, self).__init__()

        self.num_recurrent = parameters['num_recurrent']
        num_hidden = parameters['num_hidden']
        num_classes = parameters['num_classes']
        image_size = parameters['image_size']

        self.conv_stack1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        num_sheet = self.get_conv_output_size(image_size, 3, 2, 1)
        num_sheet = self.get_conv_output_size(num_sheet, 2, 0, 2)

        self.conv_stack2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        num_sheet = self.get_conv_output_size(num_sheet, 3, 2, 1)
        num_sheet = self.get_conv_output_size(num_sheet, 2, 0, 2)

        self.conv_stack3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        num_sheet = self.get_conv_output_size(num_sheet, 3, 2, 1)
        num_sheet = self.get_conv_output_size(num_sheet, 2, 0, 2)


        self.nonlinear = nn.Sequential(
            nn.Linear(self.num_recurrent, num_hidden),
            nn.ELU()
        )

        num_flattened = num_sheet * num_sheet * 128

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=parameters['dropout'])
        self.state_transition = torch.randn(self.num_recurrent, self.num_recurrent)
        self.state_transition = torch.nn.Parameter( self.state_transition / torch.max(torch.abs(torch.linalg.eig(self.state_transition)[0])) * 0.95)
        self.control_to_recurrent = nn.Linear(num_flattened + 2, self.num_recurrent)
        self.to_logits = nn.Linear(num_hidden, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, image, eye_pos, recurrent):
        '''Forward pass of the model
        Args:
            image (torch.Tensor): The input image
            eye_pos (torch.Tensor): The eye position
            recurrent (torch.Tensor): The recurrent state at the previous time step
        Returns:
            torch.Tensor: The class probabilities
            torch.Tensor: The recurrent state at the current time step
        '''

        input = self.conv_stack1(image)
        input = self.conv_stack2(input)
        input = self.conv_stack3(input)
        input = self.flatten(input)
        input = self.dropout(input)
        control = torch.cat((input, eye_pos), dim=1)
        recurrent = torch.matmul(recurrent, self.state_transition) + self.control_to_recurrent(control)
        hidden = self.nonlinear(recurrent)
        hidden = self.dropout(hidden)
        
        logits = self.to_logits(hidden)
        output = self.softmax(logits)
        return output, recurrent
    
     
    def init_recurrent(self):
        '''Initializes the recurrent state
        Args:
            batch_size (int): The batch size
        Returns:
            torch.Tensor: The initialized recurrent state
        '''

        return torch.zeros(1, self.num_recurrent)
    
    
    @staticmethod
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
    