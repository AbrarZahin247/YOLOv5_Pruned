import torch
import torch.nn as nn
from models.yolo import Model
import yaml
from utils.torch_utils import select_device,prune
from utils.general import intersect_dicts
import argparse
import os



def eval_l1_sparsity(model):
    """
    Evaluate the L1 sparsity of a PyTorch model.
    
    Args:
        model (nn.Module): The model to evaluate sparsity for.
    
    Returns:
        float: The proportion of zero (or near-zero) weights in the model.
    """
    total_weights = 0
    zero_weights = 0
    
    for param in model.parameters():
        if param.requires_grad:  # Only consider weights that require gradients (i.e., trainable weights)
            total_weights += param.numel()
            zero_weights += (param.abs() < 1e-5).sum().item()
    
    sparsity = zero_weights / total_weights
    return sparsity

def weight_transfer(original_model, pruned_model, mask):
    """
    Transfers weights from the original model to the pruned model according to the mask.
    
    Args:
        original_model (nn.Module): The original model with full weights.
        pruned_model (nn.Module): The pruned model where weights will be transferred.
        mask (dict): A dictionary where the keys are layer names and the values are masks (1 for keep, 0 for prune).
    
    Returns:
        nn.Module: The pruned model with weights transferred from the original model.
    """
    with torch.no_grad():
        for (orig_name, orig_param), (pruned_name, pruned_param) in zip(original_model.state_dict().items(), pruned_model.state_dict().items()):
            if orig_name in mask:  # Check if this layer has a corresponding mask
                # Apply the mask: copy only the unpruned weights from the original model to the pruned model
                pruned_param.data.copy_(orig_param.data * mask[orig_name])
            else:
                # If there's no mask for this layer, simply copy the weights from the original model
                pruned_param.data.copy_(orig_param.data)
    
    return pruned_model

def load_model_from_checkpoint(weight, device="cuda:0"):
    """
    Load a YOLO model from a checkpoint.
    """
    ckpt = torch.load(weight, map_location=device)
    model_cfg = ckpt['model'].yaml
    model = Model(model_cfg).to(device)
    state_dict = ckpt['model'].float().state_dict()
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=[])
    model.load_state_dict(state_dict, strict=False)
    return model, ckpt

def save_pruned_model(ckpt, new_model, save_dir, pruning_ratio):
    """
    Save the pruned model and its configuration.
    """
    model_name = f"prune_{pruning_ratio}_pruned_net.pt"
    ckpt["model"] = new_model
    ckpt["best_fitness"] = 0.0
    torch.save(ckpt, os.path.join(save_dir, model_name))

def prune_net(weight, batch_size,device,pruning_ratio,save_dir="pruned_net_saved_dir"):
    """
    Prune the YOLO model based on the L1 sparsity and save the pruned model.
    """
    try:
        device = select_device(device, batch_size=batch_size)
        # Load the model
        model, ckpt = load_model_from_checkpoint(weight, device)
        
        # Evaluate the model sparsity
        sparsity = eval_l1_sparsity(model)
        print(f"Model sparsity before pruning: {sparsity:.4f}")
        
        # Prune the model
        mask, new_cfg = prune(model, pruning_ratio)
        
        # Save the pruned model configuration
        os.makedirs(save_dir, exist_ok=True)
        cfg_name = f"prune_{pruning_ratio}_pruned_net.yaml"
        with open(os.path.join(save_dir, cfg_name), "w") as f:
            yaml.safe_dump(new_cfg, f, sort_keys=False)

        # Create and transfer weights to the pruned model
        model_pruned = Model(new_cfg).to(device)
        new_model = weight_transfer(model, model_pruned, mask)

        # Test the new model's forward pass
        test_forward_pass(new_model)

        # Save the pruned model
        save_pruned_model(ckpt, new_model, save_dir, pruning_ratio)

        print(f"Pruned model saved to {save_dir}")

    except Exception as e:
        print(f"Error during pruning: {e}")

def test_forward_pass(model, input_size=(1, 3, 640, 640), device="cuda:0"):
    """
    Test the forward pass of the model to ensure it works correctly.
    """
    model.eval()
    inputs = torch.rand(*input_size).to(device)
    outputs = model(inputs)
    print(f"Output shape from pruned model: {outputs[0].shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune YOLOv5 model")
    parser.add_argument('--weight', type=str, required=True, help='Path to YOLOv5 checkpoint.')
    parser.add_argument('--save_dir', type=str, default="pruned_net", help='Path to save output files.')
    parser.add_argument('--pruning_ratio', type=float, default=0.3, help='Pruning ratio')
    parser.add_argument('--device', default="0", help='Device to use for pruning (e.g., "cpu" or "cuda:0")')
    parser.add_argument("--batch_size", type=int, default=8, help="total batch size for all GPUs, -1 for autobatch")
    
    args = parser.parse_args()
    prune_net(weight=args.weight, save_dir=args.save_dir, pruning_ratio=args.pruning_ratio, device=args.device,batch_size=args.batch_size)

