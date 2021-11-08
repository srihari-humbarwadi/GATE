import torch
import torch.nn as nn
import torch.nn.functional as F

# assume N data domains, and T tasks, and M modalities
# flexible skeleton
# a forward pass for each modality


class DataTaskModalityAgnosticSystem(nn.Module):
    def __init__(self):
        super().__init__()
