import torch.nn as nn
import torch.nn.utils.prune as prune

def prune_channels_and_weights(model, channel_threshold, weight_threshold):
    prev_conv = None  # To keep track of the preceding Conv2d layer

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and channel_threshold is not None:
            # Channel pruning based on BatchNorm's gamma (weight) values
            mask = module.weight.abs() > channel_threshold
            pruned_channels = mask.sum().item()
            
            # Only proceed if there's a previous Conv2d layer
            if prev_conv is not None:
                # Create a new convolutional layer with fewer output channels
                new_conv = nn.Conv2d(
                    in_channels=prev_conv.in_channels,
                    out_channels=pruned_channels,
                    kernel_size=prev_conv.kernel_size,
                    stride=prev_conv.stride,
                    padding=prev_conv.padding,
                    bias=prev_conv.bias is not None
                )
                
                # Copy the non-zero weights to the new convolutional layer
                new_conv.weight.data = prev_conv.weight.data[mask, :, :, :]
                if prev_conv.bias is not None:
                    new_conv.bias.data = prev_conv.bias.data[mask]
                
                # Replace the original Conv2d layer with the pruned one
                parent_module = dict(model.named_modules())[name.rsplit('.', 2)[0]]
                setattr(parent_module, name.split('.')[-2], new_conv)

            # Replace the BatchNorm2d layer's parameters accordingly
            module.weight.data = module.weight.data[mask]
            module.bias.data = module.bias.data[mask]
            module.running_mean = module.running_mean[mask]
            module.running_var = module.running_var[mask]
            module.num_features = pruned_channels

        elif isinstance(module, nn.Conv2d):
            # Store the Conv2d layer to use for channel pruning later
            prev_conv = module
            
            # Unstructured pruning: prune individual weights within the convolutional layer
            prune.l1_unstructured(module, name="weight", amount=weight_threshold)
            
            # Remove the pruning reparametrization to make the pruning permanent
            prune.remove(module, 'weight')

    return model

