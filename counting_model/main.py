"""
Global Regression Counting with ShanghaiTech Part B (PyTorch)
-------------------------------------------------------------
This script trains a ResNet18-based regression model to predict total crowd counts
from ShanghaiTech Part B images using global regression (no density maps).

Expected directory structure:
    shanghaitech/
      part_B_train_data/
        images/
        ground_truth/
      part_B_test_data/
        images/
        ground_truth/
"""

import os, math
from pathlib import Path
import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# ============================================================
# 1️⃣  Configuration
# ============================================================

DATA_ROOT = "./shanghaitech"   # ← change to your dataset path
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 20
IMG_SIZE = (512, 512)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2️⃣  Dataset Definition
# ============================================================

class ShanghaiTechGlobalCount(Dataset):
    def __init__(self, root_dir, split="train", img_size=IMG_SIZE, transform=None):
        assert split in ("train", "test")
        self.root_dir = Path(root_dir)
        folder = "part_B_train_data" if split == "train" else "part_B_test_data"
        self.img_dir = self.root_dir / folder / "images"
        self.gt_dir  = self.root_dir / folder / "ground_truth"
        self.files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(".jpg") or f.lower().endswith(".png")
        ])
        self.img_size = img_size
        self.transform = transform or T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = str(self.img_dir / fname)
        gt_path = str(self.gt_dir / ("GT_" + os.path.splitext(fname)[0].replace("processed_","") + ".mat"))

        img = Image.open(img_path).convert("RGB")
        mat = sio.loadmat(gt_path)
        points = mat["image_info"][0,0][0,0][0]
        count = float(points.shape[0])
        img = self.transform(img)
        return img, torch.tensor(count, dtype=torch.float32)


class OpenImagesCountDataset(torch.utils.data.Dataset):
    def __init__(self, fo_view, transform=None):
        # Convert the FiftyOne view to a list to fix len()
        self.samples = list(fo_view)
        self.transform = transform or T.Compose([
            T.Resize(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample.filepath).convert("RGB")
        img = self.transform(img)
        detections = getattr(sample, "ground_truth", None)
        count = len(detections.detections) if detections else 0
        return img, torch.tensor(count, dtype=torch.float32)

# ============================================================
# 3️⃣  Model Definition
# ============================================================

class GlobalRegressorResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feat, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)

# ============================================================
# 4️⃣  Evaluation Function
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    mae = mse = n = 0
    with torch.no_grad():
        for imgs, counts in loader:
            imgs, counts = imgs.to(device), counts.to(device)
            preds = model(imgs).clamp(min=0)
            mae += (preds - counts).abs().sum().item()
            mse += ((preds - counts) ** 2).sum().item()
            n += imgs.size(0)
    return mae / n, math.sqrt(mse / n)

# ============================================================
# 5️⃣  Training Loop
# ============================================================

def train_full(data_root, train_loader, test_loader, batch_size=8, lr=1e-4, n_epochs=12):
    model = GlobalRegressorResNet(pretrained=True).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_mae = float("inf")
    train_losses = []  # ← store loss per epoch

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)  # halve lr every 5 epochs

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, counts in train_loader:
            imgs, counts = imgs.to(DEVICE), counts.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, counts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)  # ← store

        val_mae, val_rmse = evaluate(model, test_loader, DEVICE)
        print(f"Epoch {epoch:02d}: TrainLoss={train_loss:.4f} | ValMAE={val_mae:.2f} | ValRMSE={val_rmse:.2f}")
        scheduler.step()
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_resnet_global_count.pth")

    print("Training done. Best MAE:", best_mae)

    # Plot training loss
    plt.figure(figsize=(8,5))
    plt.plot(range(1, n_epochs+1), train_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss (MSE)")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.show()

    return model


# ============================================================
# 6️⃣  Visualization
# ============================================================

def visualize_examples(data_root, checkpoint="best_resnet_global_count.pth", n_examples=8):
    test_ds = ShanghaiTechGlobalCount(data_root, split="test", transform=T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225]),
    ]))
    loader = DataLoader(test_ds, batch_size=n_examples, shuffle=False)
    imgs, counts = next(iter(loader))

    model = GlobalRegressorResNet(pretrained=True).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        preds = model(imgs.to(DEVICE)).cpu().numpy()

    plt.figure(figsize=(12,6))
    for i in range(min(n_examples, imgs.size(0))):
        ax = plt.subplot(2, math.ceil(n_examples / 2), i + 1)
        img_vis = imgs[i].clone()
        img_vis = img_vis * torch.tensor([0.229,0.224,0.225]).view(3,1,1) + torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        plt.imshow(img_vis.permute(1,2,0).numpy().clip(0,1))
        plt.title(f"GT: {int(counts[i].item())}, Pred: {preds[i]:.1f}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# ============================================================
# 7️⃣  Main entry point
# ============================================================



# ============================================================
# 8️⃣ Feature Map / Activation Visualization
# ============================================================

def visualize_last_layer(data_root, checkpoint="best_resnet_global_count.pth", n_examples=4):
    model = GlobalRegressorResNet(pretrained=True).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    test_ds = ShanghaiTechGlobalCount(data_root, split="test", transform=T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225]),
    ]))
    loader = DataLoader(test_ds, batch_size=n_examples, shuffle=False)
    imgs, counts = next(iter(loader))

    # Hook to grab last conv feature maps
    features = []
    def hook_fn(module, input, output):
        features.append(output)

    handle = model.backbone.layer1.register_forward_hook(hook_fn)

    with torch.no_grad():
        preds = model(imgs.to(DEVICE)).cpu().numpy()

    handle.remove()  # unregister hook
    feature_maps = features[0].cpu()  # shape: [B, C, H, W]

    # visualize
    plt.figure(figsize=(12, 6))
    for i in range(min(n_examples, imgs.size(0))):
        ax = plt.subplot(2, n_examples, i+1)
        img_vis = imgs[i].clone()
        img_vis = img_vis * torch.tensor([0.229,0.224,0.225]).view(3,1,1) + torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        plt.imshow(img_vis.permute(1,2,0).numpy().clip(0,1))
        plt.title(f"GT: {int(counts[i].item())}, Pred: {preds[i]:.1f}")
        plt.axis("off")

        # heatmap from mean of last layer channels
        ax2 = plt.subplot(2, n_examples, n_examples + i + 1)
        fmap = feature_maps[i].mean(0)  # mean across channels
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())  # normalize 0-1
        plt.imshow(fmap.numpy(), cmap='jet', alpha=0.6)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def visualize_last_layer_temp(test_loader, checkpoint="best_resnet_global_count.pth", n_examples=4):
    model = GlobalRegressorResNet(pretrained=True).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    imgs, counts = next(iter(test_loader))

    # Hook to grab last conv feature maps
    features = []
    def hook_fn(module, input, output):
        features.append(output)

    handle = model.backbone.layer4.register_forward_hook(hook_fn)

    with torch.no_grad():
        preds = model(imgs.to(DEVICE)).cpu().numpy()

    handle.remove()  # unregister hook
    feature_maps = features[0].cpu()  # shape: [B, C, H, W]

    # visualize
    plt.figure(figsize=(12, 6))
    for i in range(min(n_examples, imgs.size(0))):
        ax = plt.subplot(2, n_examples, i+1)
        img_vis = imgs[i].clone()
        img_vis = img_vis * torch.tensor([0.229,0.224,0.225]).view(3,1,1) + torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        plt.imshow(img_vis.permute(1,2,0).numpy().clip(0,1))
        plt.title(f"GT: {int(counts[i].item())}, Pred: {preds[i]:.1f}")
        plt.axis("off")

        # heatmap from mean of last layer channels
        ax2 = plt.subplot(2, n_examples, n_examples + i + 1)
        fmap = feature_maps[i].mean(0)  # mean across channels
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())  # normalize 0-1
        plt.imshow(fmap.numpy(), cmap='jet', alpha=0.6)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_image(img_path, checkpoint, device=DEVICE):
    """
    Visualize prediction and last conv layer for a single image.
    """
    model = GlobalRegressorResNet(pretrained=True).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # shape [1,3,H,W]

    # Hook to grab last conv layer output
    features = []

    def hook_fn(module, input, output):
        features.append(output)

    handle = model.backbone.layer4.register_forward_hook(hook_fn)

    # Forward pass
    model.eval()
    with torch.no_grad():
        pred = model(img_tensor).item()

    handle.remove()
    feature_map = features[0].squeeze(0).cpu()  # shape [C,H,W]

    # Visualization
    plt.figure(figsize=(8, 4))

    # Original image
    ax1 = plt.subplot(1, 2, 1)
    img_vis = img_tensor[0].cpu().clone()
    img_vis = img_vis * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3,
                                                                                                                     1,
                                                                                                                     1)
    plt.imshow(img_vis.permute(1, 2, 0).numpy().clip(0, 1))
    plt.title(f"Predicted Count: {pred:.1f}")
    plt.axis("off")

    # Heatmap from last conv layer
    ax2 = plt.subplot(1, 2, 2)
    fmap = feature_map.mean(0)  # mean across channels
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())
    plt.imshow(fmap.numpy(), cmap='jet', alpha=0.6)
    plt.title("Last Conv Layer Activation")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_grad_cam(img_path, model, target_layer, device=DEVICE):
    """
    Visualize prediction and a Grad-CAM heatmap for a single image in a regression model.
    """
    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # shape [1, 3, H, W]

    # --- Setup for Grad-CAM ---
    # 1. Register forward hook to grab feature map (A)
    features = []

    def forward_hook_fn(module, input, output):
        features.append(output)

    # 2. Register backward hook to grab gradients (G)
    gradients = []

    def backward_hook_fn(module, grad_input, grad_output):
        # Grad_output[0] is the gradient of the loss/output w.r.t the layer's output
        gradients.append(grad_output[0])

    # Hook the target layer (model.backbone.layer4)
    # NOTE: Assuming model.backbone.layer4 is the target module
    forward_handle = target_layer.register_forward_hook(forward_hook_fn)
    backward_handle = target_layer.register_full_backward_hook(backward_hook_fn)

    # 3. Forward Pass to get prediction and feature map
    model.zero_grad()
    model.eval()
    pred = model(img_tensor)

    # 4. Backward Pass for Grad-CAM
    # Target is a regression score, so we backpropagate from the raw score (pred)
    # by using .backward(torch.ones_like(pred))
    pred.backward(torch.ones_like(pred))

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # --- Grad-CAM Calculation ---
    A = features[0].squeeze(0).cpu()  # Feature map (C, H, W)
    G = gradients[0].squeeze(0).cpu()  # Gradients (C, H, W)

    # Calculate channel importance weights (alpha_k): mean of gradients across spatial dims
    alpha = G.mean(dim=[1, 2])  # Shape (C,)

    # Weighted combination: L = ReLU(sum_k (alpha_k * A_k))
    L = torch.relu((alpha[:, None, None] * A).sum(dim=0))  # Shape (H, W)

    # Min-max normalization for visualization
    L_norm = (L - L.min()) / (L.max() - L.min())

    # --- Visualization ---
    plt.figure(figsize=(8, 4))

    # Original image (Left)
    ax1 = plt.subplot(1, 2, 1)
    # Reverse normalization for visualization
    img_vis = img_tensor[0].cpu().clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_vis = img_vis * std + mean
    plt.imshow(img_vis.permute(1, 2, 0).numpy().clip(0, 1))
    plt.title(f"Predicted Count: {pred.item():.1f}")
    plt.axis("off")

    # Grad-CAM Heatmap Overlay (Right)
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(img_vis.permute(1, 2, 0).numpy().clip(0, 1))  # Display original image first

    import torch.nn.functional as F
    # Resize heatmap to match image size for overlay
    heatmap = F.interpolate(L_norm.unsqueeze(0).unsqueeze(0),
                            size=(img_vis.shape[1], img_vis.shape[2]),
                            mode='bilinear',
                            align_corners=False).squeeze().detach().numpy()

    plt.imshow(heatmap, cmap='jet', alpha=0.6)  # Overlay heatmap
    plt.title("Grad-CAM Activation Map")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_saliency_map(img_path, model, device=DEVICE):
    """
    Generates and visualizes a Saliency Map (Input Gradient)
    to show which pixels (boundaries) most influence the prediction.
    """
    model.eval()

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # Crucially, the input tensor must require gradients
    img_tensor = transform(img).unsqueeze(0).to(device).requires_grad_(True)

    # 1. Forward Pass
    model.zero_grad()
    pred = model(img_tensor)
    predicted_count = pred.item()

    # 2. Backward Pass (Calculate Gradient)
    # Backpropagate from the single scalar prediction (y)
    # The gradient is calculated w.r.t the input tensor (img_tensor)
    pred.backward(torch.ones_like(pred))

    # 3. Extract and Process Gradients
    # The gradients w.r.t. the input (dY/dX) are stored in img_tensor.grad
    gradients = img_tensor.grad.abs().squeeze(0).cpu()  # Take absolute value, shape [C, H, W]

    # Calculate Saliency Map: Max gradient across color channels (or mean)
    # Max is often preferred to highlight the strongest influence regardless of channel
    saliency = gradients.max(dim=0)[0]  # Shape [H, W]

    # Min-max normalization for visualization
    saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    # --- Visualization ---
    plt.figure(figsize=(8, 4))

    # Original image (Left)
    ax1 = plt.subplot(1, 2, 1)
    # Reverse normalization for visual display
    img_vis = img_tensor.data.squeeze(0).cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_vis = img_vis * std + mean
    plt.imshow(img_vis.permute(1, 2, 0).numpy().clip(0, 1))
    plt.title(f"Predicted Count: {predicted_count:.1f}")
    plt.axis("off")

    # Saliency Map (Right)
    ax2 = plt.subplot(1, 2, 2)
    # Display the Saliency Map
    plt.imshow(saliency_norm.numpy(), cmap='hot')  # 'hot' or 'plasma' are good for boundaries
    plt.title("Saliency Map (Input Sensitivity)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


class GuidedReLU(torch.autograd.Function):
    """
    Custom Autograd Function implementing the Guided Backpropagation rule.
    """

    @staticmethod
    def forward(ctx, input_tensor):
        # Store the output tensor for use in the backward pass
        ctx.save_for_backward(input_tensor)
        # Standard ReLU forward pass
        return input_tensor.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the input tensor from the forward pass
        input_tensor, = ctx.saved_tensors

        # 1. Standard ReLU backprop mask (where the input was positive)
        grad_input = grad_output.clone()
        grad_input[input_tensor < 0] = 0

        # 2. Guided Backprop Rule: Only pass back positive gradients
        # Clip the negative gradients in the incoming gradient (grad_output)
        guided_grad = torch.clamp(grad_output, min=0.)

        # 3. Combine rules: Result is the standard ReLU grad mask applied to the guided gradient
        # This is where the error was previously coming from, now explicitly handled.
        grad_input[guided_grad < 0] = 0  # This line is often simplified by the above

        return guided_grad * (input_tensor > 0).float()

# 1. The Custom Autograd Function remains the same (GuidedReLU class)
# ... (GuidedReLU class definition from previous response)

# 2. Define a stateless module wrapper
class GuidedReLUModule(nn.Module):
    def forward(self, x):
        return GuidedReLU.apply(x)


def replace_relu_with_guided(module, original_modules_list):
    """
    Recursively replaces all nn.ReLU modules with the GuidedReLUModule wrapper.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            # Store original module and replace with the new nn.Module wrapper
            original_modules_list.append((module, name, child))

            # ***THE FIX***: Replace the original module with an INSTANCE of nn.Module
            setattr(module, name, GuidedReLUModule())

        else:
            # Continue recursion for other containers
            replace_relu_with_guided(child, original_modules_list)

def visualize_guided_backprop(img_path, model, device=DEVICE):
    model.eval()

    # 1. Backup and Replace all nn.ReLU with GuidedReLU
    original_modules = []

    replace_relu_with_guided(model, original_modules)  # Apply replacement to the entire model

    # --- The rest of the function remains similar ---

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose(
        [T.Resize(IMG_SIZE), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_tensor = transform(img).unsqueeze(0).to(device).requires_grad_(True)

    # 2. Forward & Backward Pass
    model.zero_grad()
    pred = model(img_tensor)
    predicted_count = pred.item()
    pred.backward(torch.ones_like(pred))

    # 3. Restore Original Modules
    for parent, name, original_module in original_modules:
        setattr(parent, name, original_module)

    # 4. Extract and Process Gradients
    gradients = img_tensor.grad.abs().squeeze(0).cpu()
    guided_map = gradients.max(dim=0)[0]
    guided_map_norm = (guided_map - guided_map.min()) / (guided_map.max() - guided_map.min() + 1e-8)

    # --- Visualization Code (omitted for brevity, assume it is the same) ---
    # ... (Visualization code from previous response)
    # ...

    # Example visualization lines:
    plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(1, 2, 1)
    # Reverse normalization and plot img_vis
    # ...
    plt.title(f"Predicted Count: {predicted_count:.1f}")

    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(guided_map_norm.numpy(), cmap='gray')
    plt.title("Guided Backpropagation (Boundaries)")
    plt.show()


if __name__ == "__main__":
    # if not os.path.exists(DATA_ROOT):
    #     raise RuntimeError(f"Dataset not found at {DATA_ROOT}. Please download and extract ShanghaiTech Part B.")
    #
    # img_size = IMG_SIZE
    # transform = T.Compose([
    #     T.Resize(img_size),
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485,0.456,0.406],
    #                 std=[0.229,0.224,0.225]),
    # ])
    #
    # train_ds = ShanghaiTechGlobalCount(DATA_ROOT, split="train", transform=transform)
    # test_ds = ShanghaiTechGlobalCount(DATA_ROOT, split="test", transform=transform)
    # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # import fiftyone as fo
    # fo.config.database_uri = "mongodb://127.0.0.1:27017"
    #
    #
    # import fiftyone.zoo as foz
    #
    # # Load the training split of OpenImages V7 for Cat and Dog classes
    # dataset = foz.load_zoo_dataset(
    #     "open-images-v7",
    #     split="train",
    #     label_types=["detections"],  # use bounding boxes
    #     classes=["Cat"],  # only these classes
    #     max_samples=50000,  # optional limit
    #     shuffle=True
    # )
    #
    # print(dataset._get_default_sample_fields)
    #
    # # Create a view with only samples that have ≥2 detections
    # F = fo.ViewField
    #
    # dataset_view = dataset.match(
    #     F("ground_truth.detections").length() >= 1
    # )
    #
    # print(f"Number of images with ≥2 objects: {len(dataset_view)}")
    #
    # # Split into 80% train, 20% test
    # n_train = int(len(dataset_view) * 0.8)
    # train_view = dataset_view.take(n_train)
    # test_view = dataset_view.skip(n_train)
    #
    # train_dataset = OpenImagesCountDataset(train_view)
    # test_dataset = OpenImagesCountDataset(test_view)
    #
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    #
    #
    # #model = train_full(DATA_ROOT, train_loader, test_loader, batch_size=BATCH_SIZE, lr=LR, n_epochs=EPOCHS)
    # #visualize_examples(DATA_ROOT)
    # visualize_last_layer_temp(test_loader, checkpoint="best_resnet_global_count.pth", n_examples=6)
    visualize_image("personal_tests/img_3.png", checkpoint="best_resnet_global_count.pth", device=DEVICE)
    visualize_image("personal_tests/img_4.png", checkpoint="best_resnet_global_count.pth", device=DEVICE)
    visualize_image("personal_tests/img_5.png", checkpoint="best_resnet_global_count.pth", device=DEVICE)
    model = GlobalRegressorResNet(pretrained=True).to(DEVICE)
    model.load_state_dict(torch.load("best_resnet_global_count.pth", map_location=DEVICE))
    model.eval()
    visualize_grad_cam("personal_tests/img_3.png", model, target_layer = model.backbone.conv1, device=DEVICE)
    visualize_grad_cam("personal_tests/img_4.png", model, target_layer=model.backbone.conv1, device=DEVICE)
    visualize_grad_cam("personal_tests/img_5.png", model, target_layer=model.backbone.conv1, device=DEVICE)

    visualize_saliency_map("personal_tests/img_3.png", model, device=DEVICE)
    visualize_saliency_map("personal_tests/img_4.png", model, device=DEVICE)
    visualize_saliency_map("personal_tests/img_5.png", model, device=DEVICE)

    visualize_guided_backprop("personal_tests/img_3.png", model, device=DEVICE)
    visualize_guided_backprop("personal_tests/img_4.png", model, device=DEVICE)
    visualize_guided_backprop("personal_tests/img_5.png", model, device=DEVICE)
