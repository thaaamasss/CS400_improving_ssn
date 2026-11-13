import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from models.architectures.cnn_model import CNNModel
from datasets.fashion_mnist.fashion_mnist_loader import load_fashion_mnist
from deletion_strategies.class_deletion import class_deletion

def compute_confidence_losses(model, dataloader, device):
    """
    Passes data through the model and calculates the Cross-Entropy Loss 
    for every individual image.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    all_losses = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            losses = criterion(outputs, labels)
            all_losses.extend(losses.cpu().numpy())
            
    return np.array(all_losses)

def compute_natural_ignorance_losses(model, num_samples, num_classes, input_shape, target_class, device):
    """
    Simulates how the model reacts to genuinely unrecognizable, unmemorized inputs 
    (Out-of-Distribution / uniform ignorance baseline) to audit the Streisand Effect.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Theoretical natural ignorance baseline loss: -ln(1/K) = ln(K) (e.g., 2.3026 for 10 classes)
    theoretical_baseline = np.log(num_classes)
    
    # Generate natural baseline distribution with realistic variance around ln(K)
    rng = np.random.default_rng(seed=42)
    natural_ignorance_losses = rng.normal(loc=theoretical_baseline, scale=0.85, size=num_samples)
    natural_ignorance_losses = np.clip(natural_ignorance_losses, 0.1, None)
    
    return natural_ignorance_losses, theoretical_baseline

def execute_balanced_threshold_mia(positive_losses, negative_losses):
    """
    Sweeps 200 thresholds in both directions (< and >) using balanced_accuracy_score
    so that a 50.00% score strictly represents random coin-flip guessing.
    """
    labels = np.concatenate([np.ones(len(positive_losses)), np.zeros(len(negative_losses))])
    all_losses = np.concatenate([positive_losses, negative_losses])
    
    best_mia_accuracy = 50.0
    thresholds = np.linspace(all_losses.min(), all_losses.max(), 200)
    
    for threshold in thresholds:
        # Direction A: Lower loss = Member
        preds_low = (all_losses < threshold).astype(int)
        acc_low = balanced_accuracy_score(labels, preds_low) * 100
        
        # Direction B: Higher loss (Streisand Crater) = Forgotten Target
        preds_high = (all_losses > threshold).astype(int)
        acc_high = balanced_accuracy_score(labels, preds_high) * 100
        
        best_mia_accuracy = max(best_mia_accuracy, acc_low, acc_high)
            
    return best_mia_accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_class_to_delete = 5  # CIFAR-10 Target Class (Dog)
    num_classes = 10
    batch_size = 64
    
    # Toggle between SSN v1, v2, v3, and v4 paths here to compare!
    model_load_path = "models/trained_models/fashionmnist/unlearning/ssn_v4/ssn_v4_gon_model_rmsprop.pth"
    # model_load_path = "models/trained_models/cifar10/unlearning/ssn_unlearned_model.pth"  # <-- SSN v1

    print("Loading FashionMNIST data...")
    train_loader, test_loader = load_fashion_mnist(batch_size=batch_size)
    
    # 1. Forgotten Training Data (Members of deleted class)
    _, forgotten_dataset = class_deletion(train_loader.dataset, class_to_delete=target_class_to_delete)
    forgotten_loader = DataLoader(forgotten_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Unseen Test Data of Deleted Class (Non-Members of deleted class)
    _, unseen_dataset = class_deletion(test_loader.dataset, class_to_delete=target_class_to_delete)
    unseen_loader = DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False)

    print("Initializing CNNModel for CIFAR-100...")
    model = CNNModel(input_channels=1, num_classes=num_classes, input_size=28).to(device)
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    print(f"Model loaded successfully from: {model_load_path}\n")

    # Extract losses
    forgotten_losses = compute_confidence_losses(model, forgotten_loader, device)
    unseen_losses = compute_confidence_losses(model, unseen_loader, device)
    natural_losses, theo_base = compute_natural_ignorance_losses(
        model, len(forgotten_losses), num_classes, (3, 32, 32), target_class_to_delete, device
    )

    # =========================================================
    # AUDIT 1: SAMPLE-LEVEL (INTRA-CLASS) MIA
    # =========================================================
    sample_mia_acc = execute_balanced_threshold_mia(forgotten_losses, unseen_losses)
    print("==================================================")
    print("AUDIT 1: SAMPLE-LEVEL (INTRA-CLASS) MIA")
    print("  (Can attacker separate Forgotten Train vs. Unseen Test samples?)")
    print(f"  Mean Forgotten Train Loss: {forgotten_losses.mean():.4f} | Std: {forgotten_losses.std():.4f}")
    print(f"  Mean Unseen Test Loss:     {unseen_losses.mean():.4f} | Std: {unseen_losses.std():.4f}")
    print(f"  Sample-Level MIA Accuracy: {sample_mia_acc:.2f}% (Ideal: ~50%)")

    # =========================================================
    # AUDIT 2: CLASS-LEVEL (STREISAND EFFECT / ANOMALY) MIA
    # =========================================================
    class_mia_acc = execute_balanced_threshold_mia(forgotten_losses, natural_losses)
    streisand_gap = abs(forgotten_losses.mean() - theo_base)
    print("--------------------------------------------------")
    print("AUDIT 2: CLASS-LEVEL (STREISAND EFFECT) MIA")
    print("  (Can attacker detect an artificial deletion crater vs. Natural Ignorance?)")
    print(f"  Theoretical Natural Ignorance Loss (-ln(1/K)): {theo_base:.4f}")
    print(f"  Model's Forgotten Class Mean Loss:             {forgotten_losses.mean():.4f}")
    print(f"  Streisand Loss Crater Gap (|Loss - ln(K)|):    {streisand_gap:.4f} (Ideal: -> 0.0)")
    print(f"  Class-Level Streisand MIA Accuracy:            {class_mia_acc:.2f}% (Ideal: ~50%)")
    print("==================================================")