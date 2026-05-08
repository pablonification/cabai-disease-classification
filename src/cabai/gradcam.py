import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def get_target_layer(model):
    """
    Heuristic to find the target layer for Grad-CAM based on the model architecture.
    Supports timm efficientnet, torchvision efficientnet, and resnet.
    """
    # timm EfficientNet
    if hasattr(model, 'conv_head'):
        return [model.conv_head]
    # torchvision EfficientNet
    elif hasattr(model, 'features'):
        return [model.features[-1]]
    # ResNet
    elif hasattr(model, 'layer4'):
        return [model.layer4[-1]]
    else:
        # Fallback: try to just return the last layer that is a module
        # This might not always work, but acts as a safeguard
        children = list(model.children())
        if len(children) > 0:
            return [children[-1]]
        raise ValueError("Could not automatically determine target layer for Grad-CAM.")

def generate_gradcam(model, input_tensor, rgb_img, target_category=None):
    """
    Generate a Grad-CAM visualization for a given image.
    
    Args:
        model: The trained PyTorch model.
        input_tensor: The preprocessed image tensor (1, C, H, W).
        rgb_img: The original image as a numpy array with float values in [0, 1].
        target_category: The class index to generate the CAM for. If None, uses the highest scoring class.
        
    Returns:
        cam_image: The original image with the heatmap overlaid (numpy array in [0, 1]).
    """
    target_layers = get_target_layer(model)
    
    # We construct the CAM object once. In a real app we might cache this, 
    # but constructing it per inference is fast enough for demo.
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # If target_category is None, pytorch-grad-cam automatically uses the highest scoring class
        targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
        
        # Generate the grayscale CAM (batch size 1, so we take [0])
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Overlay the heatmap on the original image
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
    return cam_image
