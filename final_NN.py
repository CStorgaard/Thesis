import torch
import numpy as np

class AlsingActivation(torch.nn.Module):
    def __init__(self, num_features):
        super(AlsingActivation, self).__init__()
        self.beta = torch.nn.Parameter(torch.ones(num_features))
        self.gamma = torch.nn.Parameter(torch.ones(num_features) * 0.01)

    def forward(self, x):
        sigmoid_part = torch.sigmoid(self.beta * x)
        return (self.gamma + (1.0 - self.gamma) * sigmoid_part) * x

class DRMD_Emulator(torch.nn.Module):
    def __init__(self, input_dim=3, output_dim=25, hidden_nodes=128):
        super(DRMD_Emulator, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_nodes),
            AlsingActivation(hidden_nodes),
            torch.nn.Linear(hidden_nodes, hidden_nodes),
            AlsingActivation(hidden_nodes),
            torch.nn.Linear(hidden_nodes, hidden_nodes),
            AlsingActivation(hidden_nodes),
            torch.nn.Linear(hidden_nodes, output_dim) 
        )
        
        # Create placeholder buffers. load_state_dict() will seamlessly overwrite 
        # these with the actual means and standard deviations from your training!
        self.register_buffer('target_mean', torch.zeros(output_dim))
        self.register_buffer('target_scale', torch.ones(output_dim))

    def forward(self, x):
        scaled_output = self.network(x)
        # The model automatically un-scales its own predictions!
        reconstructed_spectra = (scaled_output * self.target_scale) + self.target_mean
        return reconstructed_spectra

# =====================================================================

def get_emulator_prediction(z, fidm, dNeff, checkpoint_path="alsing_emulator_v1.pt"):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Initialize the model
    model = DRMD_Emulator()
    
    # Load the trained weights AND the target buffers (NO strict=False here!)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    a_val = 1.0 / (1.0 + z)
    raw_inputs = np.array([[fidm, dNeff, a_val]])
    
    scaled_inputs = (raw_inputs - checkpoint['input_scaler_mean']) / checkpoint['input_scaler_scale']
    input_tensor = torch.tensor(scaled_inputs, dtype=torch.float32)
    
    with torch.no_grad(): 
        pred_Ck = model(input_tensor).squeeze(0).numpy()
        
    return checkpoint['k_grid'], pred_Ck