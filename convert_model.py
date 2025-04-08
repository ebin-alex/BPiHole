import torch
import torch.nn as nn
import os
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MAMLModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MAMLModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def convert_model(input_model_path, output_model_path, input_size=9, hidden_size=64, output_size=1):
    """
    Convert a trained model to the format expected by our system
    
    Args:
        input_model_path: Path to the input model file
        output_model_path: Path to save the converted model
        input_size: Number of input features
        hidden_size: Size of hidden layer
        output_size: Number of output classes
    """
    try:
        # Check if input model exists
        if not os.path.exists(input_model_path):
            logging.error(f"Input model file '{input_model_path}' not found")
            return False
            
        # Load the input model
        input_model = torch.load(input_model_path, map_location=torch.device('cpu'))
        
        # Initialize our model architecture
        model = MAMLModel(input_size, hidden_size, output_size)
        
        # If the input model is a state dict, load it directly
        if isinstance(input_model, dict):
            model.load_state_dict(input_model)
        else:
            # If it's a full model, extract the state dict
            model.load_state_dict(input_model.state_dict())
            
        # Save the model state dict
        torch.save(model.state_dict(), output_model_path)
        logging.info(f"Model converted and saved to '{output_model_path}'")
        return True
        
    except Exception as e:
        logging.error(f"Error converting model: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert a trained model to the format expected by our system')
    parser.add_argument('--input', type=str, required=True, help='Path to the input model file')
    parser.add_argument('--output', type=str, default='relation_net.pth', help='Path to save the converted model')
    parser.add_argument('--input_size', type=int, default=9, help='Number of input features')
    parser.add_argument('--hidden_size', type=int, default=64, help='Size of hidden layer')
    parser.add_argument('--output_size', type=int, default=1, help='Number of output classes')
    
    args = parser.parse_args()
    
    success = convert_model(
        args.input, 
        args.output, 
        args.input_size, 
        args.hidden_size, 
        args.output_size
    )
    
    if success:
        print("Model conversion completed successfully")
    else:
        print("Model conversion failed")

if __name__ == "__main__":
    main() 