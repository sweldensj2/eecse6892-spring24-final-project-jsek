import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN_improved(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DQN_improved, self).__init__()
        
        # Define the neural network architecture
        self.model = nn.Sequential(
            nn.Linear(num_inputs, 256),  # Input layer
            nn.ReLU(),                    # Activation function
            nn.BatchNorm1d(256),          # Batch normalization
            nn.Dropout(0.5),              # Dropout for regularization
            
            nn.Linear(256, 512),          # Hidden layer
            nn.ReLU(),                    # Activation function
            nn.BatchNorm1d(512),          # Batch normalization
            nn.Dropout(0.5),              # Dropout for regularization
            
            nn.Linear(512, 256),          # Hidden layer
            nn.ReLU(),                    # Activation function
            nn.BatchNorm1d(256),          # Batch normalization
            nn.Dropout(0.5),              # Dropout for regularization
            
            nn.Linear(256, num_outputs)   # Output layer
        )
        
        # Initialize the weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        # Forward pass through the network
        return self.model(x)
    
    def get_action(self, input):
        qvalues = self.forward(input)
        action = torch.argmax(qvalues, dim=1)
        return action.cpu().numpy()
