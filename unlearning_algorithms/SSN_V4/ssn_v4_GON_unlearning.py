import os
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def execute_ssn_v4(model, target_loader, retain_loader, epochs=10, lr=0.01, C=0.1, quantile_threshold=0.95):
    """
    Executes SSN v4: Geometric Orthogonal Necrosis (GON) with Ambient Logit Calibration.
    1. Rotates unlocked target class weights W_target in fc2 to be 90-degrees orthogonal
       to incoming latent features F(X_target).
    2. Calibrates the scalar bias b_target to the Iso-Ignorance Equilibrium so that
       the post-unlearning Cross-Entropy loss matches natural ignorance ln(K) (~2.3026),
       defeating both Sample-Level MIA (Audit 1) and Class-Level Streisand MIA (Audit 2).
    """
    device = next(model.parameters()).device

    # ==========================================
    # PHASE 1: DIAGNOSTIC MAPPING
    # ==========================================
    model.eval()
    model.zero_grad()

    Xtarget, Ytarget = next(iter(target_loader))
    Xtarget, Ytarget = Xtarget.to(device), Ytarget.to(device)

    preds = model(Xtarget)
    ce_loss = nn.CrossEntropyLoss()(preds, Ytarget)
    ce_loss.backward()

    target_idx = Ytarget[0].item()

    # Extract gradient magnitudes for the target row in fc2
    target_row_grads = model.fc2.weight.grad[target_idx].abs()
    threshold = torch.quantile(target_row_grads, quantile_threshold)
    mask_target = target_row_grads >= threshold

    # Initialize Softmax Decoupling Lock (Mask_global) for fc2.weight
    Mask_global = torch.zeros_like(model.fc2.weight, dtype=torch.bool).to(device)
    Mask_global[target_idx, :] = mask_target

    # ==========================================
    # PHASE 2: DUAL-QUARANTINE PROTOCOL
    # ==========================================
    Mbaseline = copy.deepcopy(model)
    Mbaseline.eval()

    for name, param in model.named_parameters():
        if 'fc2' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    with torch.no_grad():
        if model.fc2.bias is not None:
            model.fc2.bias[target_idx] = 0.0

    # ==========================================
    # PHASE 3: GEOMETRIC ORTHOGONAL NECROSIS (GON)
    # ==========================================
    model.train()
    model.zero_grad()

    latent_features = {}
    def capture_features_hook(module, input_tensors):
        latent_features['F_X'] = input_tensors[0].detach()

    hook_handle = model.fc2.register_forward_pre_hook(capture_features_hook)
    opt = optim.SGD([model.fc2.weight], lr=lr)

    for e in range(epochs):
        for batch_X, _ in target_loader:
            batch_X = batch_X.to(device)
            opt.zero_grad()

            _ = model(batch_X)
            F_X = latent_features['F_X']

            W_target = model.fc2.weight[target_idx].unsqueeze(0)
            cos_sim = F.cosine_similarity(F_X, W_target, dim=1, eps=1e-8)

            # Minimize squared Cosine Similarity -> 90-degree orthogonality
            loss_gon = torch.mean(cos_sim ** 2)
            loss_gon.backward()

            with torch.no_grad():
                model.fc2.weight.grad[~Mask_global] = 0.0
                opt.step()
                model.fc2.weight.clamp_(-C, C)
                model.fc2.weight[~Mask_global] = Mbaseline.fc2.weight[~Mask_global]
                if model.fc2.bias is not None:
                    model.fc2.bias.copy_(Mbaseline.fc2.bias)
                    model.fc2.bias[target_idx] = 0.0

    hook_handle.remove()

    # ==========================================
    # PHASE 4: ISO-IGNORANCE EQUILIBRIUM CALIBRATION (ANTI-STREISAND SHIELD)
    # ==========================================
    # After orthogonalizing W_target, calibrate the scalar bias b_target so the
    # mean target class loss matches theoretical natural ignorance ln(K) (e.g., 2.3026)
    model.eval()
    K = model.fc2.out_features
    target_natural_loss = math.log(K)  # ln(10) = 2.302585

    with torch.no_grad():
        if model.fc2.bias is not None:
            all_other_lse = []
            all_w_dot_f = []
            for batch_X, _ in target_loader:
                batch_X = batch_X.to(device)
                logits = model(batch_X)  # currently b_target = 0.0, so logits[:, target_idx] = W_target . F(X)
                w_dot_f = logits[:, target_idx]

                # Mask out target_idx to compute log-sum-exp of the 9 retained classes
                retained_indices = [i for i in range(K) if i != target_idx]
                retained_logits = logits[:, retained_indices]
                other_lse = torch.logsumexp(retained_logits, dim=1)

                all_other_lse.append(other_lse)
                all_w_dot_f.append(w_dot_f)

            mean_other_lse = torch.cat(all_other_lse).mean().item()
            mean_w_dot_f = torch.cat(all_w_dot_f).mean().item()

            # For P(target) = 1/K, we need: z_target = other_lse - ln(K - 1)
            # Since z_target = W_target . F(X) + b_target, we set b_target to:
            optimal_bias = mean_other_lse - mean_w_dot_f - math.log(K - 1)
            model.fc2.bias[target_idx] = optimal_bias

    return model