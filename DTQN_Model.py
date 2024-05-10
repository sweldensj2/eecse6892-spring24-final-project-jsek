import torch
import torch.nn as nn
import torch.nn.functional as F

class DTQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DTQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 64)
        self.Tlayer = nn.TransformerEncoderLayer(d_model=64, nhead=2)
        self.transformerE = nn.TransformerEncoder(self.Tlayer, num_layers=3)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.fc(x)
        out = self.transformerE(x)
        out = F.relu(self.fc1(out))
        qvalue = self.fc2(out)

        return qvalue
    
    def get_action(self, input):
        input = input.unsqueeze(0)
        qvalue = self.forward(input)
        action = torch.argmax(qvalue)
        return action.cpu().numpy()