import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torchvision.models.alexnet import AlexNet_Weights
import os

class ClassifierPipelineAlexnet:
    def __init__(self, model_name='alexnet', num_classes=12, predict_img_dir='./predicted_images'):
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = ['उनी', 'काम', 'घर', 'छ', 'त्यो', 'नेपाली', 'म', 'मेरो', 'रुख', 'शिक्षक', 'साथी', 'हो']
        self.model = None
        self.predict_img_dir = predict_img_dir
        self.prediction_counts = {class_name: 0 for class_name in self.class_names}

        if not os.path.exists(self.predict_img_dir):
            os.makedirs(self.predict_img_dir)

    def initialize_model(self):
        torch.cuda.empty_cache()
        self.model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.model.classifier[6] = nn.Linear(4096, self.num_classes)
        self.model.classifier.add_module('7', nn.LogSoftmax(dim=1))
        self.model = self.model.to(self.device)

    def load_model(self, path='./models/trained_model_alex01.pth'):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model = self.model.to(self.device)

    def predict(self, img, img_path):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        try:
            image = Image.fromarray(img).convert('L')
            if image.size[0] == 0 or image.size[1] == 0:
                return None

            image = transform(image).unsqueeze(0)
            image = image.to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
                _, preds = torch.max(outputs, 1)

            predicted_class = self.class_names[preds.item()]
            self.prediction_counts[predicted_class] += 1

            save_path = os.path.join(self.predict_img_dir, f"{predicted_class}_{os.path.basename(img_path)}")
            Image.fromarray(img).save(save_path)

            return predicted_class
        except Exception as e:
            return None

if __name__ == "__main__":
    pipeline = ClassifierPipelineAlexnet()
    pipeline.initialize_model()
    pipeline.load_model()
