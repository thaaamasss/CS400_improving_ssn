import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

def execute_ssn_v3(model, target_loader, retain_loader, epochs=15, lr=0.005, C=0.5, T=15.0):
    """
    Executes the complete SSN v3 unlearning pipeline using Temperature-Scaled KL-Divergence.
    """
    device = next(model.parameters()).device
    
    # ---------------------------------------------------------
    # PHASE 1: DIAGNOSTIC MAPPING
    # ---------------------------------------------------------
    model.eval()
    model.zero_grad()
    
    Xtarget, Ytarget = next(iter(target_loader))
    Xtarget, Ytarget = Xtarget.to(device), Ytarget.to(device)
    
    preds = model(Xtarget)
    ce_loss = nn.CrossEntropyLoss()(preds, Ytarget)
    ce_loss.backward()
    
    target_idx = Ytarget[0].item()
    
    target_row_grads = model.fc2.weight.grad[target_idx].abs()
    mask_target = target_row_grads > target_row_grads.mean()
    
    Mask_global = torch.zeros_like(model.fc2.weight, dtype=torch.bool).to(device)
    Mask_global[target_idx, :] = mask_target

    # ---------------------------------------------------------
    # PHASE 2: DUAL-QUARANTINE PROTOCOL
    # ---------------------------------------------------------
    Mbaseline = copy.deepcopy(model)
    Mbaseline.eval()
    
    for name, param in model.named_parameters():
        if 'fc2' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # ---------------------------------------------------------
    # PHASE 3: NECROTIC EXECUTION (KL-DIVERGENCE SMOOTHING)
    # ---------------------------------------------------------
    model.train()
    
    # FIX: Injected momentum to push past local probability spikes
    opt = optim.SGD(model.fc2.parameters(), lr=lr, momentum=0.9)
    
    K = model.fc2.out_features 
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    for e in range(epochs):
        for batch_X, _ in target_loader:
            batch_X = batch_X.to(device)
            batch_size = batch_X.size(0)
            
            uniform_target = torch.full((batch_size, K), 1.0 / K).to(device)
            
            opt.zero_grad()
            preds_necrotic = model(batch_X)
            
            # FIX: Apply Temperature Scaling to melt logit rigidity
            soft_logits = preds_necrotic / T
            log_probs = F.log_softmax(soft_logits, dim=1)
            
            loss_necrotic = kl_loss_fn(log_probs, uniform_target)
            
            # FIX: Scale gradients by T^2 to compensate for the division
            loss_necrotic = loss_necrotic * (T * T)
            loss_necrotic.backward()
            
            with torch.no_grad():
                model.fc2.weight.grad[~Mask_global] = 0.0
                opt.step()
                model.fc2.weight.clamp_(-C, C)
                model.fc2.weight[~Mask_global] = Mbaseline.fc2.weight[~Mask_global]
                
    return model