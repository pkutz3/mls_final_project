import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets
import json
import torchvision.models as models
from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchattacks
import matplotlib.pyplot as plt

# Custom Dataset Wrapper
class PreprocessedImageNetV2(torch.utils.data.Dataset):
    def __init__(self, variant="matched-frequency"):
        self.dataset = ImageNetV2Dataset(variant)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.preprocess(img), label

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def __call__(self, x, class_idx=None):
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Grad-CAM calculation
        weights = F.adaptive_avg_pool2d(self.gradients, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.squeeze().cpu().detach()

# 2. Get correctly classified samples
def get_correct_samples(model, dataloader, num_samples=1000):
    correct_images = []
    correct_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            correct_mask = preds == labels
            correct_images.append(images[correct_mask])
            correct_labels.append(labels[correct_mask])
            
            if len(torch.cat(correct_images)) >= num_samples:
                break
    
    return (torch.cat(correct_images)[:num_samples], 
            torch.cat(correct_labels)[:num_samples])

# Get GradCAM visualizations
def get_gradcam_visuals(model, images, labels, gradcam):
    heatmaps = []
    for img, label in zip(images, labels):
        heatmap = gradcam(img.unsqueeze(0), label.unsqueeze(0))
        heatmaps.append(heatmap)
    return torch.stack(heatmaps)

#  Quantitative analysis
def calculate_entropy(heatmap):
    # Flatten the heatmap and compute entropy
    heatmap_flattened = heatmap.flatten(1)
    
    # Ensure no zero values in the heatmap for probability calculation
    heatmap_flattened = heatmap_flattened + 1e-8  # Add small epsilon to avoid log(0)

    # Normalize the heatmap to get the probability distribution
    prob_distribution = heatmap_flattened / heatmap_flattened.sum(dim=1, keepdim=True)
    
    # Ensure no values are zero after normalization
    prob_distribution = torch.clamp(prob_distribution, min=1e-8)
    
    # Compute entropy
    entropy = -torch.sum(prob_distribution * torch.log(prob_distribution + 1e-8), dim=1)
    
    # Return the mean entropy
    return entropy.mean().item()

def analyze_heatmap_changes(clean_heatmaps, adv_heatmaps):
    # Pixel-wise difference
    diffs = (clean_heatmaps - adv_heatmaps).abs().mean(dim=[1,2])
    
    # Attention shift metrics
    clean_max = clean_heatmaps.flatten(1).max(dim=1).indices
    adv_max = adv_heatmaps.flatten(1).max(dim=1).indices
    pos_shift = (clean_max != adv_max).float().mean()
    
    # Entropy calculation
    clean_entropy = calculate_entropy(clean_heatmaps)
    adv_entropy = calculate_entropy(adv_heatmaps)
    entropy_diff = adv_entropy - clean_entropy
    
    # Print results
    print(f"Mean absolute difference: {diffs.mean().item():.4f}")
    print(f"Attention position changed: {pos_shift.item()*100:.2f}% of cases")
    print(f"Clean Heatmap Entropy: {clean_entropy:.4f}")
    print(f"Adversarial Heatmap Entropy: {adv_entropy:.4f}")
    print(f"Entropy Difference: {entropy_diff:.4f}")

# Get ImageNet class names
with open('imagenet_class_index.json') as f:
    class_idx = json.load(f)

imagenet_classes = [class_idx[i] for i in range(1000)]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and dataset
    model = models.resnet18(pretrained=True).to(device).eval()
    dataset = PreprocessedImageNetV2("matched-frequency")
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

    correct_images, correct_labels = get_correct_samples(model, dataloader)

    # 1. Setup GradCAM
    target_layer = model.layer4[-1]  # Last conv layer for ResNet18
    gradcam = GradCAM(model, target_layer)

    # 3. Generate adversarial attacks on correct samples
    pgd_attack = torchattacks.PGD(
        model,
        eps=4/255,       # Maximum perturbation (8 pixel intensity)
        alpha=2/255,     # Attack step size
        steps=10,        # Number of attack iterations
        random_start=True # Start from random point in epsilon ball
    )
    adv_images = pgd_attack(correct_images.clone(), correct_labels.clone())

    clean_heatmaps = get_gradcam_visuals(model, correct_images, correct_labels, gradcam)
    adv_heatmaps = get_gradcam_visuals(model, adv_images, correct_labels, gradcam)

    analyze_heatmap_changes(clean_heatmaps, adv_heatmaps)

