import torch
import torch.nn as nn
import onnx
import onnxruntime
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversion.log'),
        logging.StreamHandler()
    ]
)

class RelationNetwork(nn.Module):
    def __init__(self, input_size=15, hidden_size=64):
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
        emb1 = self.embed(x1)
        emb2 = self.embed(x2)
        combined = torch.cat((emb1, emb2), dim=-1)
        relation_score = self.relation_layer(combined)
        return relation_score

def convert_to_onnx(model_path, output_path):
    """Convert PyTorch model to ONNX format"""
    try:
        # Load the PyTorch model with correct input size
        model = RelationNetwork(input_size=15, hidden_size=64)  # Changed to 15 features
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Create dummy inputs for both feature vector and reference vector
        dummy_input1 = torch.randn(1, 15)  # Feature vector with 15 features
        dummy_input2 = torch.randn(1, 15)  # Reference vector with 15 features
        
        # Export the model
        torch.onnx.export(model,               # model being run
                         (dummy_input1, dummy_input2),  # model input (tuple for multiple inputs)
                         output_path,          # where to save the model
                         export_params=True,   # store the trained parameter weights inside the model file
                         opset_version=11,    # the ONNX version to export the model to
                         do_constant_folding=True,  # whether to execute constant folding for optimization
                         input_names=['input1', 'input2'],   # the model's input names
                         output_names=['output'], # the model's output names
                         dynamic_axes={'input1': {0: 'batch_size'},    # variable length axes
                                     'input2': {0: 'batch_size'},
                                     'output': {0: 'batch_size'}})
        
        logging.info(f"Model converted and saved to {output_path}")

        # Verify the model can be loaded
        ort_session = onnxruntime.InferenceSession(output_path)
        
        # Run a test inference
        ort_inputs = {
            'input1': dummy_input1.numpy(),
            'input2': dummy_input2.numpy()
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        logging.info("Model verified successfully with ONNX Runtime")
        
        return True
        
    except Exception as e:
        logging.error(f"Error converting model: {str(e)}")
        return False

if __name__ == "__main__":
    # Try both model paths
    model_paths = [
        "Model/relation_net.pth",  # Highest accuracy model
        "training/models/relation_net.pt"  # Backup model
    ]
    
    output_dir = "Model"
    os.makedirs(output_dir, exist_ok=True)
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            output_path = os.path.join(output_dir, "blackhole_detector.onnx")
            logging.info(f"Converting model from {model_path}")
            if convert_to_onnx(model_path, output_path):
                logging.info("Conversion successful!")
                break
            else:
                logging.warning(f"Failed to convert {model_path}, trying next model if available") 