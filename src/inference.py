import torch
from torchvision.transforms import ToTensor, Grayscale, Resize, Compose
from PIL import Image
from architecture import *
import os
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load('src/model.pth', map_location=device, weights_only=False)
    model.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    return model, device

class ImageDataset(Dataset):
    def __init__(self, folder_path):
        self.image_paths = []
        self.filenames = []
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                self.image_paths.append(os.path.join(root, file))
                self.filenames.append(file)
                    
        
        self.transform = Compose([
            Grayscale(num_output_channels=1),
            Resize((20, 20)),
            ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        filename = self.filenames[idx]
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, filename


def process_folder_images_in_batches(folder_path, model, batch_size, device='cpu'):
    predictions = {}

    if not os.path.isdir(folder_path):
        print(f"Error: The folder {folder_path} does not exist.")
        return predictions

    dataset = ImageDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)

    with torch.no_grad():
        model.eval()
        
        for images, filenames in dataloader:
            images = images.to(device)

            outputs = model(images)

            _, predicted_classes = torch.max(outputs, 1)

            for i in range(len(filenames)):
                predictions[filenames[i]] = predicted_classes[i].item()

    json_filename = os.path.join(folder_path, 'predictions.json')
    with open(json_filename, 'w') as json_file:
        json.dump(predictions, json_file, indent=4)

    return predictions


def classify_all(dict):
    model, device = load_model()
    for k, v in tqdm(dict.items(), desc="Running Inference"):
        folder_path = 'temp/' + str(v) 
        process_folder_images_in_batches(folder_path, model, batch_size=64, device=device)

