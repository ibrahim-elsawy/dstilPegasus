from typing import Dict, List, Union

from catalyst.core import MetricCallback
import torch
from torch import nn
from torch.nn import functional as F



class CrossentropylossCallback(MetricCallback):
    """
    CosineLossCallback
    This callback is calculating cosine loss between hidden states
    of the two hugging face transformers models.
    """

    def __init__(
        self,
        input_key: Union[str, List[str], Dict[str, str]] = None,
        output_key: Union[str, List[str], Dict[str, str]] = None,
        prefix: str = "cross_ent_loss",
        multiplier: float = 1.0,
        **metric_kwargs,
    ):
        """
        Args:
            input_key (Union[str, List[str], Dict[str, str]]): key/list/dict
                of keys that takes values from the input dictionary
                If '__all__', the whole input will be passed to the criterion
                If None, empty dict will be passed to the criterion.
            output_key (Union[str, List[str], Dict[str, str]]): key/list/dict
                of keys that takes values from the input dictionary
                If '__all__', the whole output will be passed to the criterion
                If None, empty dict will be passed to the criterion.
            prefix (str): prefix for metrics and output key for loss
                in ``state.batch_metrics`` dictionary
            criterion_key (str): A key to take a criterion in case
                there are several of them and they are in a dictionary format.
            multiplier (float): scale factor for the output loss.
        """
        if output_key is None:
            output_key = [
                "target",
                "s_logits",
            ]
        super().__init__(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
            metric_fn=self.metric_fn,
            **metric_kwargs,
        )

    def metric_fn(
        self,
        target: torch.Tensor,
        s_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes label smoothing loss on given hidden states
        Args:
            labes: tensor from dataset labels
            s_logits: tensor from student model
        Returns:
            label smothing loss
        """
        pad_token_id = 0
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        loss = loss_fct(s_logits.view(-1, s_logits.shape[-1]), target.view(-1))
        return loss
