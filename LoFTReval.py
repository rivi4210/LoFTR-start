import torch
import time
import torchvision.transforms as transforms
from kornia.feature import LoFTR
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from performance import PerformanceChecker

transform = transforms.Compose([
    transforms.Resize((320, 240)),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor()
])


def load_image(image_path):
    """Load an image from file and apply transformations."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension


def load_loftr(device, pretrained='outdoor', weights_path="/notebooks/weights/outdoor.ckpt"):
    """Load the LoFTR model with the specified pretrained weights."""
    if pretrained not in ['outdoor', 'indoor_new', 'indoor', None]:
        raise ValueError(f"pretrained should be None or one of ['outdoor', 'indoor_new', 'indoor']")
    model = LoFTR(pretrained=None)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device)['state_dict'])
    else:
        print(f"Downloading pretrained weights for {pretrained} model...")
        model = LoFTR(pretrained=pretrained)

    model.to(device).eval()  # Ensure the model is on the correct device
    return model


def match_images(img1, img2, model, device):
    """Match features between two images using LoFTR."""
    with torch.no_grad():
        img1 = img1.to(device)
        img2 = img2.to(device)

        if img1.dim() != 4 or img2.dim() != 4:
            raise ValueError(f"Expected 4D input, but got shapes {img1.shape} and {img2.shape}")

        matches = model({'image0': img1, 'image1': img2})
        print(f"Number of matches: {len(matches['keypoints0'])}")
    return matches


def visualize_matches(img1, img2, matches):
    """Visualize the matching results with lines connecting matching keypoints."""
    img1 = transforms.ToPILImage()(img1.squeeze(0).cpu())
    img2 = transforms.ToPILImage()(img2.squeeze(0).cpu())

    keypoints0 = matches['keypoints0'].cpu().numpy()
    keypoints1 = matches['keypoints1'].cpu().numpy()

    img1_width, img1_height = img1.size
    img2_width, img2_height = img2.size
    combined_img = Image.new('RGB', (img1_width + img2_width, max(img1_height, img2_height)))

    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1_width, 0))

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.imshow(combined_img)

    ax.scatter(keypoints0[:, 0], keypoints0[:, 1], color='blue', s=10, label='Image 1 Keypoints')
    ax.scatter(keypoints1[:, 0] + img1_width, keypoints1[:, 1], color='green', s=10, label='Image 2 Keypoints')

    ax.axvline(x=img1_width, color='white', linewidth=2)

    for i in range(len(keypoints0)):
        x0, y0 = keypoints0[i]
        x1, y1 = keypoints1[i]
        ax.plot([x0, x1 + img1_width], [y0, y1], color='black', linewidth=1)

    ax.legend()
    plt.show()


def ablation_study():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(PerformanceChecker.check_gpu())

    # Define configurations for ablation study
    pretrained_options = {
        'outdoor': "/notebooks/weights/outdoor.ckpt",  # Original config
        None: None  # No pretrained weights
    }

    for pretrained, weights_path in pretrained_options.items():
        print(f"Testing configuration: pretrained={pretrained}")

        model = load_loftr(device, pretrained=pretrained, weights_path=weights_path)

        # Load your images from local paths
        img1_path = './images/image3.jpg'  # Replace with your image path
        img2_path = './images/image4.jpg'  # Replace with your image path
        img1 = load_image(img1_path).to(device)
        img2 = load_image(img2_path).to(device)

        matches = match_images(img1, img2, model, device)

        inference_time = PerformanceChecker.measure_inference_time(model, {'image0': img1, 'image1': img2}, device)
        print(inference_time)

        gpu_performance = PerformanceChecker.check_gpu_performance_and_memory(model, [(img1, img2)])
        print("GPU Performance metrics:")
        print(f"GPU Time: {gpu_performance['GPU Time (s)']} seconds")
        print(f"GPU Memory Usage: {gpu_performance['GPU Memory Usage (MB)']} MB")
        print(f"Peak GPU Memory Usage: {gpu_performance['Peak GPU Memory Usage (MB)']} MB")

        visualize_matches(img1.squeeze(0).cpu(), img2.squeeze(0).cpu(), matches)
        break


if __name__ == "__main__":
    ablation_study()
