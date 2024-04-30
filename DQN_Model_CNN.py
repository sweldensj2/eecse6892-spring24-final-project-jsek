import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN_CNN(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(DQN_CNN, self).__init__()
        self.num_outputs = num_outputs

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_shape[2], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=22528, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_outputs)

    def forward(self, x):
        # reshape x from the state [210, 160, 1] --> [1, 210, 160]
        # print("x before", x.shape, )
        if len(x.shape) == 3:  # Check if the tensor has three dimensions
            x = torch.unsqueeze(x, 0)  # Unsqueeze the tensor to add a batch dimension

        x = x.permute(0, 3, 1, 2)
        # print("x after", x.shape)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output of the convolutional layers
        # print("x before flattening", x.shape)
        x = x.reshape(x.size(0), -1)
        # print("x after flattening", x.shape)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_action(self, input):
        qvalues = self.forward(input)
        action = torch.argmax(qvalues, dim=1)
        return action.cpu().numpy()
