"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for CQ-CNN
================================================================
Provides visual explainability for both Classical and Quantum hybrid networks.
Computes activation heatmaps over MRI slices to show anatomical regions
(e.g., hippocampus, ventricles, cortical atrophy) contributing to diagnosis.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class GradCAM:
    """
    Grad-CAM engine for PyTorch convolutional models (ClassicalNet / QuantumNet).
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else model.conv2
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=1):
        """
        Computes Grad-CAM heatmap for a single input image tensor (1, 1, H, W).
        
        Args:
            input_tensor: torch.Tensor of shape (1, 1, H, W)
            target_class: integer index (0=Non-Demented, 1=Moderate Dementia)
            
        Returns:
            heatmap: numpy 2D array of shape (H, W) normalized to [0, 1]
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Output shape is (1, 2)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks failed to capture activations/gradients.")

        # Global average pooling of gradients
        # activations: (1, C, H_feat, W_feat), gradients: (1, C, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to input dimensions
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def apply_gradcam_overlay(image_pil, heatmap, colormap="jet", alpha=0.45):
    """
    Overlays a Grad-CAM heatmap onto a grayscale PIL MRI image.

    Args:
        image_pil: PIL.Image in grayscale ("L") or RGB
        heatmap: 2D numpy array [0, 1]
        colormap: matplotlib colormap name ('jet', 'inferno', 'magma', 'viridis')
        alpha: heatmap transparency (0.0 to 1.0)

    Returns:
        overlaid_img: PIL.Image (RGB)
    """
    orig_w, orig_h = image_pil.size
    img_gray = np.array(image_pil.convert("L").resize((orig_w, orig_h)), dtype=np.float32) / 255.0

    # Resize heatmap to match original PIL image size
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.BILINEAR)
    heatmap_norm = np.array(heatmap_img, dtype=np.float32) / 255.0

    # Colorize heatmap (supports both new and older matplotlib)
    try:
        import matplotlib as mpl
        cmap = mpl.colormaps[colormap]
    except Exception:
        cmap = cm.get_cmap(colormap)
    colored_heatmap = cmap(heatmap_norm)[:, :, :3]  # drop alpha channel (H, W, 3)

    # Base RGB image
    base_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)

    # Blend
    blended = (1.0 - alpha) * base_rgb + alpha * colored_heatmap
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


if __name__ == "__main__":
    print("Testing GradCAM module...")
    from scirpts.track_b_training import ClassicalNet

    model = ClassicalNet(conv2_ch=4)
    cam = GradCAM(model, target_layer=model.conv2)
    fake_input = torch.randn(1, 1, 128, 128, requires_grad=True)

    heatmap = cam.generate_heatmap(fake_input, target_class=1)
    print(f"Heatmap generated successfully! Shape: {heatmap.shape}, Range: [{heatmap.min():.2f}, {heatmap.max():.2f}]")

    test_img = Image.fromarray(np.random.randint(0, 255, (128, 128), dtype=np.uint8), mode="L")
    overlay = apply_gradcam_overlay(test_img, heatmap)
    print(f"Overlay created successfully! Size: {overlay.size}, Mode: {overlay.mode}")
    print("GradCAM self-test passed!")
