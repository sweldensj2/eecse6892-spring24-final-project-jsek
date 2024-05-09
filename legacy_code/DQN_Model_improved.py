import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN_improved(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DQN_improved, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        
        self.fc1 = nn.Linear(num_inputs, 512)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(128, num_outputs)

        # Dropout layers
        
        
        
        
        # Initialize the weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        # Forward pass through the network

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)

        return x
        
    
    def get_action(self, input):
        qvalue = self.forward(input)
        action = torch.argmax(qvalue)
        return action.cpu().numpy()
