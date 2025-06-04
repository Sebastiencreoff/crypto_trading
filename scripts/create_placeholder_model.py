import torch
import torch.nn as nn
import os

# Define the PlaceholderNet class (identical to the one in ai_algo.py)
class PlaceholderNet(nn.Module):
    def __init__(self, input_size=5, output_size=3): # Assuming sequence length 5
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    # Ensure the 'models' directory exists
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    # Define the model path
    model_path = os.path.join(output_dir, "ai_algo_model.pth")

    # Create an instance of the model
    model = PlaceholderNet()

    # Save the model's state dictionary
    # For saving the entire model (including architecture), use torch.save(model, model_path)
    # For saving only state_dict (weights), use torch.save(model.state_dict(), model_path)
    # ai_algo.py uses torch.load(model_path) which expects the entire model.
    torch.save(model, model_path)

    print(f"Placeholder model saved to {model_path}")
