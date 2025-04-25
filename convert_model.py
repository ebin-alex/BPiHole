import os
import torch
import logging
import argparse
from torch import nn

class RelationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relation_layer = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def embed(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

    def forward(self, x1, x2):
        # Get embeddings for both inputs
        x1_embed = self.embed(x1)
        x2_embed = self.embed(x2)
        # Concatenate embeddings
        combined = torch.cat((x1_embed, x2_embed), dim=-1)
        # Get relation score
        relation_score = self.relation_layer(combined)
        return relation_score

def convert_model(input_path, output_path, input_size=13, hidden_size=64):
    """
    Convert a PyTorch model to TorchScript format
    """
    try:
        # Load the PyTorch model
        model = RelationNetwork(input_size=input_size, hidden_size=hidden_size)
        model.load_state_dict(torch.load(input_path))
        model.eval()

        # Create dummy inputs (batch_size=1)
        dummy_input1 = torch.randn(1, input_size)
        dummy_input2 = torch.randn(1, input_size)

        # Test the model with dummy inputs
        with torch.no_grad():
            test_output = model(dummy_input1, dummy_input2)
            logging.info(f"Test output shape: {test_output.shape}")

        # Export the model to TorchScript
        scripted_model = torch.jit.script(model)
        scripted_model.save(output_path)

        # Verify the model
        loaded_model = torch.jit.load(output_path)
        with torch.no_grad():
            output = loaded_model(dummy_input1, dummy_input2)
            logging.info(f"Model successfully converted and verified: {output_path}")
            logging.info(f"Verification output shape: {output.shape}")
            return True

    except Exception as e:
        logging.error(f"Error converting model: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to TorchScript')
    parser.add_argument('--input', type=str, required=True, help='Path to input PyTorch model')
    parser.add_argument('--output', type=str, required=True, help='Path to output TorchScript model')
    parser.add_argument('--input_size', type=int, default=13, help='Input feature size')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size')
    
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Convert the model
    success = convert_model(
        args.input,
        args.output,
        input_size=args.input_size,
        hidden_size=args.hidden_size
    )

    if not success:
        logging.error("Model conversion failed")
        exit(1)

if __name__ == "__main__":
    main() 