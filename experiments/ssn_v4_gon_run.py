import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.architectures.cnn_model import CNNModel
from datasets.fashion_mnist.fashion_mnist_loader import load_fashion_mnist
from deletion_strategies.class_deletion import class_deletion
from unlearning_algorithms.SSN_V4.ssn_v4_GON_unlearning import execute_ssn_v4
from evaluation.plot_single_run import generate_run_plots

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total if total > 0 else 0.0

if __name__ == "__main__":
    # --- RUN CONFIGURATION ---
    dataset_name = "FashionMNIST"
    model_name = "RMSProp"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_class_to_delete = 5
    batch_size = 64

    model_load_path = "models/trained_models/fashionmnist/learning/rmsprop_model.pth"
    # Isolated v4 save path so v1, v2, and v3 models remain untouched
    model_save_path = "models/trained_models/fashionmnist/unlearning/ssn_v4/ssn_v4_gon_model_rmsprop.pth"

    print(f"Loading {dataset_name} data and isolating classes...")
    train_loader, test_loader = load_fashion_mnist(batch_size=batch_size)
    base_train_dataset = train_loader.dataset
    retain_dataset, target_dataset = class_deletion(base_train_dataset, class_to_delete=target_class_to_delete)

    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

    print("Initializing CNNModel for FashionMNIST...")
    model = CNNModel(input_channels=1, num_classes=10, input_size=28).to(device)

    if os.path.exists(model_load_path):
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        print(f"Baseline model loaded successfully from {model_load_path}.")
    else:
        print(f"Warning: Baseline model not found at {model_load_path}.")

    print("\n--- BASELINE SANITY CHECK ---")
    base_target = evaluate_accuracy(model, target_loader, device)
    base_retain = evaluate_accuracy(model, retain_loader, device)
    print(f"Pre-Unlearning Target Class Accuracy:   {base_target:.2f}%")
    print(f"Pre-Unlearning Retained Classes Accuracy: {base_retain:.2f}%")
    print("-----------------------------\n")

    # ==========================================
    # TIMER START & SSN v4 EXECUTION
    # ==========================================
    start_time = time.time()
    print("Initiating SSN v4 (Geometric Orthogonal Necrosis - GON)...")

    unlearned_model = execute_ssn_v4(
        model=model,
        target_loader=target_loader,
        retain_loader=retain_loader,
        epochs=5,
        lr=0.01,
        C=0.1,
        quantile_threshold=0.85  # Use 0.95 for CIFAR-10/100, 0.85 for MNIST/FashionMNIST
    )

    end_time = time.time()
    execution_time = end_time - start_time
    # ==========================================
    # TIMER END
    # ==========================================

    print("SSN v4 (GON) Unlearning Complete.\n")
    print(f"=== SSN v4 Execution Time: {execution_time:.4f} seconds ===\n")

    print("Evaluating SSN v4 Isolation and Destruction...")
    target_acc = evaluate_accuracy(unlearned_model, target_loader, device)
    retain_acc = evaluate_accuracy(unlearned_model, retain_loader, device)
    print(f"Target Class Accuracy:   {target_acc:.2f}% (Goal: ~0-10% Natural Dispersion)")
    print(f"Retained Classes Accuracy: {retain_acc:.2f}% (Goal: Intact, close to baseline)")

    # ==========================================
    # NEW: GEOMETRIC ORTHOGONALITY & DISTRIBUTION VERIFICATION
    # ==========================================
    print("\n--- GON ORTHOGONALITY & PROBABILITY VERIFICATION ---")
    unlearned_model.eval()
    with torch.no_grad():
        sample_inputs, _ = next(iter(target_loader))
        sample_input = sample_inputs[0:1].to(device)
        raw_output = unlearned_model(sample_input)
        probabilities = F.softmax(raw_output, dim=-1)

        print(f"Raw Logit for Target Class {target_class_to_delete}: {raw_output[0, target_class_to_delete].item():.4f} (Goal: ~0.0000)")
        print("Post-Unlearning Prediction Probabilities for a Single Target Image:")
        for i, prob in enumerate(probabilities[0]):
            print(f"Class {i}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
    print("----------------------------------------------------\n")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(unlearned_model.state_dict(), model_save_path)
    print(f"GON model saved to {model_save_path}")

    # ==========================================
    # GENERATE PLOTS
    # ==========================================
    print("\nGenerating evaluation plots...")
    final_probs_array = probabilities[0].cpu().numpy()
    generate_run_plots(
        dataset_name=dataset_name,
        model_name=f"{model_name} (SSN v4 GON)",
        exec_time=execution_time,
        pre_target=base_target,
        pre_retain=base_retain,
        post_target=target_acc,
        post_retain=retain_acc,
        target_probs=final_probs_array
    )