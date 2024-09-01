import torch
import time
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from kornia.feature import LoFTR
from PIL import Image
# from performance import PerformanceChecker
from timer import Timer, print_memory_usage
from LoFTR.src.config import default
from sklearn.metrics import normalized_mutual_info_score
# from skimage.metrics import normalized_mutual_info_score
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Tuple, Union, Optional

cfg = default.get_cfg_defaults()

transform = transforms.Compose([
    transforms.Resize((500, 500), interpolation=cv2.INTER_LANCZOS4),  
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor()
])

def load_image(image_path: str) -> torch.Tensor:
    """Load an image from file and apply transformations."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def load_loftr(device: str, pretrained: Optional[str] = 'outdoor', weights_path: str = "/notebooks/weights/outdoor.ckpt") -> LoFTR:
    """Load the LoFTR model with the specified pretrained weights."""
    if pretrained not in ['outdoor', 'indoor_new', 'indoor', None]:
        raise ValueError(f"pretrained should be None or one of ['outdoor', 'indoor_new', 'indoor']")
    model = LoFTR(pretrained=None)
    model.match_type = cfg.LOFTR.MATCH_COARSE.MATCH_TYPE

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device)['state_dict'])
    else:
        print(f"Downloading pretrained weights for {pretrained} model...")
        model = LoFTR(pretrained=pretrained)
        
    model.to(device).eval()  # Ensure the model is on the correct device
    return model

def match_images(img1: torch.Tensor, img2: torch.Tensor, model: LoFTR, device: str, timer: Timer) -> Dict[str, torch.Tensor]:
    """Match features between two images using LoFTR."""
    with torch.no_grad():
        img1 = img1.to(device)
        img2 = img2.to(device)
        
        if img1.dim() != 4 or img2.dim() != 4:
            raise ValueError(f"Expected 4D input, but got shapes {img1.shape} and {img2.shape}")
        
        timer.start_timer()
        matches = model({'image0': img1, 'image1': img2})
        timer.end_timer()
        
        threshold = cfg.LOFTR.MATCH_COARSE.THR
        keypoints0 = matches['keypoints0'].cpu().numpy()
        keypoints1 = matches['keypoints1'].cpu().numpy()
        confidences = matches.get('confidence', torch.ones(len(keypoints0))).cpu().numpy()
        
        valid_matches = confidences >= threshold
        keypoints0 = keypoints0[valid_matches]
        keypoints1 = keypoints1[valid_matches]
        
        print(f"Number of matches after thresholding: {len(keypoints0)}")
        
    return {'keypoints0': torch.tensor(keypoints0), 'keypoints1': torch.tensor(keypoints1), 'confidences': confidences[valid_matches]}


def normalize_image(image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Normalize image to [0, 1]."""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if np.max(image) > 1:
        image = image / 255.0  # Normalize to [0, 1] if values are in [0, 255]
    return image

def calculate_nmi(image1: Union[np.ndarray, torch.Tensor], image2: Union[np.ndarray, torch.Tensor]) -> float:
    """Calculate Normalized Mutual Information (NMI) between two images."""
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)
    # Convert images to grayscale if they are RGB
    if image1.ndim == 3 and image1.shape[0] == 3:
        image1 = np.mean(image1, axis=0)
    if image2.ndim == 3 and image2.shape[0] == 3:
        image2 = np.mean(image2, axis=0)
    # Compute histograms
    hist1, _ = np.histogram(image1, bins=256, range=(0, 1))
    hist2, _ = np.histogram(image2, bins=256, range=(0, 1))
    # Normalize histograms
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    # Compute NMI
    nmi = normalized_mutual_info_score(hist1, hist2)
    return nmi

# def save_registered_image(image: np.ndarray, save_path: str) -> None:
#     """Convert image to 8-bit and save the registered image to the specified path."""
#     if isinstance(image, np.ndarray):
#         # Normalize the image to 0-255 and convert to uint8
#         image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         image = Image.fromarray(image)
#     image.save(save_path)
#     print(f"Registered image saved to {save_path}")

def save_registered_image(image: np.ndarray, save_path: str) -> None:
    """Convert image to 8-bit and save the registered image to the specified path."""
    if isinstance(image, np.ndarray):
        # Normalize the image to 0-255 and convert to uint8
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image = Image.fromarray(image)
    
    # Create a unique filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_save_path = f"{save_path}_{timestamp}.png"
    
    image.save(unique_save_path)
    print(f"Registered image saved to {unique_save_path}")

def create_transformation_matrix(image1: Union[torch.Tensor, np.ndarray], image2: Union[torch.Tensor, np.ndarray], keypoints1: np.ndarray, keypoints2: np.ndarray) -> np.ndarray:
    """Create a homography matrix to transform image2 to align with image1."""
    if isinstance(image1, torch.Tensor):
        image1 = image1.squeeze().cpu().numpy()  # Remove batch and channel dimensions, move to CPU and convert to numpy array
    if isinstance(image2, torch.Tensor):
        image2 = image2.squeeze().cpu().numpy()  # Remove batch and channel dimensions, move to CPU and convert to numpy array

    # Ensure images are in the correct format
    if image1.ndim == 2:
        height, width = image1.shape
    elif image1.ndim == 3:
        height, width = image1.shape[:2]
    else:
        raise ValueError("Unsupported image shape.")

    # Convert keypoints to numpy arrays if they are not already
    keypoints1 = np.array(keypoints1)
    keypoints2 = np.array(keypoints2)
    
    H, status = cv2.findHomography(keypoints2, keypoints1, cv2.RANSAC, 5.0)
    height, width = image1.shape[:2]
    registered_image2 = cv2.warpPerspective(image2, H, (width, height))
    return registered_image2

def convert_to_np(tensor_image):
    """Convert a tensor image to a NumPy array if it is a PyTorch tensor."""
    if isinstance(tensor_image, torch.Tensor):
        return tensor_image.squeeze().cpu().numpy()
    elif isinstance(tensor_image, np.ndarray):
        return tensor_image.squeeze()  # No need to convert if already a NumPy array
    else:
        raise TypeError("Input must be a PyTorch tensor or a NumPy array")


def compute_ssim(img1, img2, data_range):
    """Compute the SSIM between two images."""
    img1_np = convert_to_np(img1)
    img2_np = convert_to_np(img2)
    
    # Ensure images are within the valid range
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)
    
    # Ensure images are at least 7x7 in size
    if img1_np.shape[0] < 7 or img1_np.shape[1] < 7:
        raise ValueError('Image dimensions are too small for SSIM computation.')

    # Adjust window size based on image dimensions
    min_size = min(img1_np.shape[:2])
    win_size = min(min_size, 7)  # Use the smaller dimension or 7, whichever is smaller

    # Compute SSIM with the appropriate window size and channel_axis
    return ssim(img1_np, img2_np, multichannel=False, win_size=win_size, channel_axis=-1, data_range=data_range)

# def check_image_range(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     min_val = np.min(image)
#     max_val = np.max(image)
#     print(f"Min pixel value: {min_val}")
#     print(f"Max pixel value: {max_val}")

def calculate_statistics(num_matches, match_confidence, memory, time, nmi, ssim):
    nmi_np = np.array(nmi)    
    ssim_np = np.array(ssim)
    match_confidence_np = np.array(match_confidence)

    stats = {
        "NMI": {
            "mean": np.mean(nmi_np).round(4),
            "median": np.median(nmi_np).round(4),
            "min": np.min(nmi_np).round(4),
            "max": np.max(nmi_np).round(4)
        },
        "SSIM": {
            "mean": np.mean(ssim_np).round(4),
            "median": np.median(ssim_np).round(4),
            "min": np.min(ssim_np).round(4),
            "max": np.max(ssim_np).round(4)
        },
        "Match Confidence": {
            "mean": np.mean(match_confidence_np).round(4),
            "median": np.median(match_confidence_np).round(4),
            "min": np.min(match_confidence_np).round(4),
            "max": np.max(match_confidence_np).round(4)
        }
    }
    
    return stats
    

def ablation_study() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    csv_path = '/notebooks/image_pairs.csv'
    df = pd.read_csv(csv_path)

    timer = Timer()
    
    # Define configurations for ablation study
    pretrained_options = {
        'outdoor': "/notebooks/weights/outdoor.ckpt",  # Original config
        None: None  # No pretrained weights
    }
    
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)  # Create the results directory if it doesn't exist
    
    num_matches_lst = []
    match_confidence_lst = []
    memory_lst = []
    time_lst = []
    nmi_lst = []
    ssim_lst = []
    
    for index, row in df.iterrows():
        img1_path = f"{row['query_name']}"
        img2_path = f"{row['ref_name']}"
        
        for pretrained, weights_path in pretrained_options.items():
            print(f"Testing configuration: pretrained={pretrained}")

            model = load_loftr(device, pretrained=pretrained, weights_path=weights_path)
            
            img1 = load_image(img1_path).to(device)
            img2 = load_image(img2_path).to(device)

            matches = match_images(img1, img2, model, device, timer)
            num_matches_lst.append((len(matches['keypoints0'])))
            # print(f"Number of matches {len(matches['keypoints0'])}")
            
            match_confidence = matches['confidences'].mean().item()
            match_confidence_lst.append(match_confidence)
            # print(f"Match confidence {match_confidence}")
            
            # print_memory_usage()
            time_lst.append(timer.get_elapsed_time())
            # print(f"Time it took {timer.get_elapsed_time()}")
            
            registered_img2 = create_transformation_matrix(img1, img2, matches['keypoints0'], matches['keypoints1'])
            registered_img_save_path = f"{results_dir}/registered_img_{pretrained}.png"
            save_registered_image(registered_img2, registered_img_save_path)

            nmi = calculate_nmi(img1, registered_img2)
            nmi_lst.append(nmi)
            print(f"Normalized Mutual Information (NMI) {nmi:.4f}")
            
            # check_image_range('./images/img3.png')
            ssim_score = compute_ssim(img1, registered_img2, data_range=255)
            ssim_lst.append(ssim_score)
            print(f"SSIM Score: {ssim_score:.4f}")

            break
    print(calculate_statistics(num_matches_lst, match_confidence_lst, memory_lst, time_lst, nmi_lst, ssim_lst))

if __name__ == "__main__":
    ablation_study()
