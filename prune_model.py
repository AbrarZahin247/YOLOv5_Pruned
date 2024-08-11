import torch.nn.utils.prune as prune
import torch.nn as nn

def prune_channels_and_weights(model, channel_threshold=1e-3, weight_threshold=0.5):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            # Channel pruning based on BatchNorm's gamma (weight) values
            mask = module.weight.abs() > channel_threshold
            pruned_channels = mask.sum().item()
            
            # Create a new convolutional layer with fewer channels
            new_conv = nn.Conv2d(
                in_channels=int(module.weight.shape[0]),
                out_channels=pruned_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=False
            )
            
            # Copy the non-zero weights to the new convolutional layer
            new_conv.weight.data = module.weight.data[mask]
            if module.bias is not None:
                new_conv.bias.data = module.bias.data[mask]
            new_conv.out_channels = pruned_channels
            
            # Replace the original layer with the pruned one
            module = new_conv

        elif isinstance(module, nn.Conv2d):
            # Unstructured pruning: prune individual weights within the convolutional layer
            prune.l1_unstructured(module, name="weight", amount=weight_threshold)
            
            # Remove the pruning reparametrization to make the pruning permanent
            prune.remove(module, 'weight')

    return model
