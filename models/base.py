import torch
import torch.nn as nn
import torch.nn.functional as F


# assume N data domains, and T tasks, and M modalities
# flexible skeleton
# a forward pass for each modality
# fprop for each modality_embeddings
# tasks might involve one or multiple modalities
# their update might inmvolve multiple modality
# how the hell do we define an optimization scheme such that we can
# explicit optimizer for each modality?

# Task:
# modalities, output shape, task process (losses etc)

# Data domains:
# modalities, different distributions

# Modalities:
# A learning system (e.g. pretrained imagenet model that can be fine tuned,
# or a meta-learned MAML etc).
# Then gate should be able to evaluate the learning process on previously unseen tasks
# some will allow some training, and some will not (zero-shot)
#

class DataTaskModalityAgnosticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = nn.Module()

    def forward_images(self, x):
        return self.resnet.forward(x)

    def forward_audio(self, x):
        # b, c, sequence_length)
        return

    def forward_video(self, x):
        return NotImplementedError(f'Video forward pass not implemented in model: '
                                   f'{self.__class__.__name__}')

    def forward_text(self, x):
        return NotImplementedError(f'Text forward pass not implemented in model: '
                                   f'{self.__class__.__name__}')

    def optimizer(self):
        return NotImplementedError(f'Optimizer not implemented in model: '
                                   f'{self.__class__.__name__}')

    def train_step(self, x, y):
        return NotImplementedError(f'Train step not implemented in model: '
                                   f'{self.__class__.__name__}')

    def eval_step(self, x, y):
        return NotImplementedError(f'Evaluation step not implemented in model: '
                                   f'{self.__class__.__name__}')


class ResNetModel(nn.Module):
    def __init__(self, modalities_supported: set):
        super().__init__()

        # ('image', 'audio',
        #  'text', 'image_audio', 'image_text')
        self.audio_head = nn.Module()
        self.resnet = nn.Module()

    def forward_images(self, x):
        logits, features = self.resnet.forward(x)
        return logits, features

    def forward_dual_inputs(self, x):
        # (b, c, sequence_length)
        return

    def forward_video(self, x):
        return NotImplementedError(f'Video forward pass not implemented in model: '
                                   f'{self.__class__.__name__}')

    def forward_text(self, x):
        return NotImplementedError(f'Text forward pass not implemented in model: '
                                   f'{self.__class__.__name__}')

    def optimizer(self):
        return NotImplementedError(f'Optimizer not implemented in model: '
                                   f'{self.__class__.__name__}')

    def train_step(self, x, y):
        return NotImplementedError(f'Train step not implemented in model: '
                                   f'{self.__class__.__name__}')

    def eval_step(self, x, y):
        return NotImplementedError(f'Evaluation step not implemented in model: '
                                   f'{self.__class__.__name__}')

    def forward(self, x, y):
        pass
