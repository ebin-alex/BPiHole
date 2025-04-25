import torch
import torch.nn as nn
import onnx
import tensorflow as tf

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
        combined = torch.cat((x1, x2), dim=-1)
        relation_score = self.relation_layer(combined)
        return relation_score

def main():
    # Load PyTorch model
    input_size = 15  # Number of features
    hidden_size = 64
    model = RelationNetwork(input_size, hidden_size)
    model.load_state_dict(torch.load('models/relation_net.pt'))
    model.eval()

    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, input_size)
    dummy_proto = torch.randn(1, input_size)

    # Export to ONNX
    torch.onnx.export(model,
                     (dummy_input, dummy_proto),
                     'models/model.onnx',
                     input_names=['input', 'prototype'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                 'prototype': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}})

    # Convert ONNX to TensorFlow
    onnx_model = onnx.load('models/model.onnx')
    tf_rep = tf2onnx.convert.from_onnx(onnx_model)
    tf_model = tf.keras.models.load_model(tf_rep)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    tflite_model = converter.convert()

    # Save TFLite model
    with open('models/model.tflite', 'wb') as f:
        f.write(tflite_model)

if __name__ == '__main__':
    main() 