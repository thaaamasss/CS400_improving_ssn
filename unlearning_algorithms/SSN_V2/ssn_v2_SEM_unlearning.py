import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def execute_ssn_unlearning(model, target_loader, retain_loader, epochs=5, lr=0.01):
    device = next(model.parameters()).device
    model.train() 

    target_class = next(iter(target_loader))[1][0].item()

    model.zero_grad()
    criterion = nn.CrossEntropyLoss()

    # 1. Diagnostic pass to get gradients (CE is maintained here to identify target features)
    for inputs, labels in target_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        break 

    parameter_masks = {}
    baseline_weights = {}
    all_grads = []

    # Restrict mapping STRICTLY to the final classification head (fc2)
    for name, param in model.named_parameters():
        if 'fc2' in name and param.requires_grad and param.grad is not None:
            all_grads.append(param.grad.abs().view(-1))

    # Calculate 85th percentile threshold
    if len(all_grads) > 0:
        concatenated_grads = torch.cat(all_grads)
        threshold = torch.quantile(concatenated_grads, 0.85)
    else:
        threshold = 0.0

    # 2. Apply Boundary Quarantine & Softmax Lock
    for name, param in model.named_parameters():
        if 'fc2' in name and param.requires_grad and param.grad is not None:
            mask = param.grad.abs() >= threshold
            
            # The "Softmax Decoupling Lock": Protect the retained class rows
            if 'weight' in name:
                valid_row = mask[target_class, :].clone()
                mask[:, :] = False
                mask[target_class, :] = valid_row
            elif 'bias' in name:
                valid_val = mask[target_class].clone()
                mask[:] = False
                mask[target_class] = valid_val
                
            parameter_masks[name] = mask
        else:
            # FIXED: The "Latent Space Lock" - physically freeze gradients
            param.requires_grad = False 
            parameter_masks[name] = torch.zeros_like(param, dtype=torch.bool)
            
        baseline_weights[name] = param.clone().detach()

    model.zero_grad()
    
    # FIXED: Initialize dumb SGD to bypass Adam's memory.
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )

    # 3. Shannon Entropy Maximization (The Unlearning Loop)
    for epoch in range(epochs):
        # Labels are not needed for calculating entropy
        for inputs, _ in target_loader:
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Convert raw logits to probabilities and log-probabilities
            probs = F.softmax(outputs, dim=-1)
            log_probs = F.log_softmax(outputs, dim=-1)
            
            # Shannon Entropy formula: H(P) = -sum(p * log(p))
            entropy = -torch.sum(probs * log_probs, dim=-1).mean()
            
            # Objective: Maximize Entropy -> Minimize Negative Entropy
            loss = -1.0 * entropy
            loss.backward()
            
            # Mask out gradients for the locked portions of fc2
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None and name in parameter_masks:
                        param.grad[~parameter_masks[name]] = 0.0
                        
            optimizer.step()

            # 4. Apply Anti-Swamping Clamps and Absolute State Lock
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in parameter_masks:
                        mask = parameter_masks[name]
                        if mask.any():
                            # The Clamp
                            clamped_param = torch.clamp(param, min=-0.1, max=0.1)
                            param.copy_(torch.where(mask, clamped_param, baseline_weights[name]))
                        else:
                            # The Absolute State Lock for frozen weights
                            param.copy_(baseline_weights[name])

    return model