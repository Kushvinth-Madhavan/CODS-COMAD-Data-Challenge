import os
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, CLIPModel, CLIPProcessor

# Configurations
IMAGES_DIR = "/kaggle/input/fashion-product-images-dataset/"
STYLES_DIR = "/kaggle/input/fashion-product-images-dataset/styles/"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class FashionDatasetCLIPViT(Dataset):
    def _init_(self, images_dir, styles_dir):
        self.images_dir = images_dir
        self.styles_dir = styles_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _len_(self):
        return len(self.image_files)

    def _getitem_(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        json_path = os.path.join(self.styles_dir, image_file.replace('.jpg', '.json'))

        image = Image.open(image_path).convert("RGB")
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        text = metadata.get("data", {}).get("productDisplayName", "")

        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
        return inputs

# Model setup
class ViT_CLIP_Model(nn.Module):
    def _init_(self):
        super(ViT_CLIP_Model, self)._init_()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.fc = nn.Linear(768 + 512, 10)  # Output for 10 attributes

    def forward(self, inputs):
        image_features = self.vit(pixel_values=inputs['pixel_values']).last_hidden_state.mean(dim=1)
        text_features = self.clip.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        combined = torch.cat((image_features, text_features), dim=1)
        output = self.fc(combined)
        return output

# Initialize dataset, dataloader, model, optimizer, etc.
dataset = FashionDatasetCLIPViT(IMAGES_DIR, STYLES_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = ViT_CLIP_Model().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
def train_model(num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = torch.randint(0, 10, (inputs['input_ids'].size(0),), device=DEVICE)  # Dummy labels

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Training loop (same as previous example)
train_model()
