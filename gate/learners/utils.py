import torch
import torch.nn.functional as F

from gate.base.utils.loggers import get_logger

log = get_logger(__name__, set_default_handler=False)


def learning_scheduler_smart_autofill(
    lr_scheduler_config, num_train_samples, batch_size
):
    """
    This function is used to autofill the learning scheduler config options.
    """
    if lr_scheduler_config["_target_"].split(".")[-1] == "CosineAnnealingLR":
        if "T_max" not in lr_scheduler_config:
            lr_scheduler_config["T_max"] = num_train_samples / batch_size
    elif (
        lr_scheduler_config["_target_"].split(".")[-1]
        == "CosineAnnealingWarmRestarts"
    ):
        if "T_0" not in lr_scheduler_config:
            lr_scheduler_config["T_0"] = num_train_samples / batch_size // 2

    elif lr_scheduler_config["_target_"].split(".")[-1] == "ReduceLROnPlateau":
        lr_scheduler_config["patience"] = (
            lr_scheduler_config["patience"] * torch.cuda.device_count()
            if torch.cuda.is_available()
            else 1
        )

    return lr_scheduler_config


def get_num_samples(targets, num_classes, dtype=None):
    batch_size = targets.size(0)
    with torch.no_grad():
        # log.info(f"Batch size is {batch_size}")
        ones = torch.ones_like(targets, dtype=dtype)
        # log.info(f"Ones tensor is {ones.shape}")
        num_samples = ones.new_zeros((batch_size, num_classes))
        # log.info(f"Num samples tensor is {num_samples.shape}")
        num_samples.scatter_add_(1, targets, ones)
    return num_samples


def get_prototypes(embeddings, targets, num_classes):
    """Compute the prototypes (the mean vector of the embedded training/support
    points belonging to its class) for each classes in the task.

    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor
        has shape `(batch_size, num_examples, embedding_size)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has
        shape `(batch_size, num_examples)`.

    num_classes : int
        Number of classes in the task.

    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    """
    batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)

    num_samples = get_num_samples(targets, num_classes, dtype=embeddings.dtype)
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    prototypes = embeddings.new_zeros(
        (batch_size, num_classes, embedding_size)
    )
    indices = targets.unsqueeze(-1).expand_as(embeddings)
    prototypes.scatter_add_(1, indices, embeddings).div_(num_samples)

    return prototypes


def prototypical_loss(prototypes, embeddings, targets, **kwargs):
    """Compute the loss (i.e. negative log-likelihood) for the prototypical
    network, on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.

    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(batch_size, num_examples, embedding_size)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(batch_size, num_examples)`.

    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
    """
    squared_distances = torch.sum(
        (prototypes.unsqueeze(2) - embeddings.unsqueeze(1)) ** 2, dim=-1
    )
    return F.cross_entropy(-squared_distances, targets, **kwargs)


def get_accuracy(prototypes, embeddings, targets):
    """Compute the accuracy of the prototypical network on the test/query points.
    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(meta_batch_size, num_examples)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    sq_distances = torch.sum(
        (prototypes.unsqueeze(1) - embeddings.unsqueeze(2)) ** 2, dim=-1
    )
    _, predictions = torch.min(sq_distances, dim=-1)
    return torch.mean(predictions.eq(targets).float())
