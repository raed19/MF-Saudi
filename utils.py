import torch
import numpy as np
import transformers
import numpy as np

# Function to convert a spectrogram to an image for visualization
def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

def setlr(optimizer, lr):
    """
    Set the learning rate of the optimizer.

    Args:
        optimizer: The optimizer object.
        lr (float): The new learning rate.

    Returns:
        optimizer: The updated optimizer object.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def lr_decay(optimizer, epoch, initial_lr):
    """
    Apply learning rate decay at specified epochs.

    Args:
        optimizer: The optimizer object.
        epoch (int): The current epoch.
        initial_lr (float): The initial learning rate.

    Returns:
        optimizer: The updated optimizer object.
    """
    if epoch % 20 == 0:
        new_lr = initial_lr / (10**(epoch // 20))
        optimizer = setlr(optimizer, new_lr)
        print(f'Changed learning rate to {new_lr}')
    return optimizer

def download_pretrained_model(pretrained_model_name):
    """
    Downloads a pre-trained model and tokenizer based on the specified model name.

    Args:
        pretrained_model_name (str): Name of the pre-trained model.

    Returns:
        tokenizer: The pre-trained tokenizer.
        model: The pre-trained model.
    """
    if pretrained_model_name == 'asafaya/bert-base-arabic':
        # Load tokenizer and model for 'asafaya/bert-base-arabic'
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'asafaya/bert-base-arabic')
        model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'asafaya/bert-base-arabic')
    elif pretrained_model_name == "aubmindlab/bert-base-arabertv01":
        # Load tokenizer and model for 'aubmindlab/bert-base-arabertv01'
        tokenizer = transformers.AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv01" ,do_lower_case=False)
        model = transformers.AutoModel.from_pretrained("aubmindlab/bert-base-arabertv01")
    else:
        # Handle case where model name is not recognized
        raise ValueError(f"Cannot find model for name: {pretrained_model_name}")

    return tokenizer, model

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled
