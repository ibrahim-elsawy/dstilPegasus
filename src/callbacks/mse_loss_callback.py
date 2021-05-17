from typing import Dict, List, Union
from torch.nn import functional as F
from catalyst.core import MetricCallback
import torch
from torch import nn


LAYERS_TO_SUPERVISE = {
    # maps  num layers in student -> which teacher layers to copy.
    6: {1: [5], 2: [3, 5], 3: [1, 4, 5], 4: [1, 2, 4, 5]},
    12: {1: [11], 2: [5, 11], 3: [3, 7, 11], 6: [1, 3, 5, 8, 10, 11]},
    16: {1: [15], 3:[0, 8, 15], 4: [4, 9, 12, 15], 8: [1, 3, 5, 7, 9, 11, 13, 15],9: [0, 1, 3, 5, 7, 9, 11, 13, 15]},
}

def get_layers_to_supervise(n_student, n_teacher) -> List[int]:
    """Used or the --supervise_forward kwarg"""
    if n_student > n_teacher:
        raise ValueError(f"Cannot perform intermediate supervision for student {n_student} > teacher {n_teacher}")
    elif n_teacher == n_student:
        return list(range(n_teacher))
    elif n_student == 1:
        return [n_teacher - 1]
    else:
        return LAYERS_TO_SUPERVISE[n_teacher][n_student]

def calc_hidden_loss(attention_mask, hidden_states, hidden_states_T, matches, normalize_hidden):
        """MSE(student_hid, teacher_hid[matches]). Called "Intermediate supervision" in paper. Inspired by TinyBERT."""
        msg = "expected list or tuple for hidden_states, got tensor of shape: "
        assert not isinstance(hidden_states, torch.Tensor), f"{msg}{hidden_states.shape}"
        assert not isinstance(hidden_states_T, torch.Tensor), f"{msg}{hidden_states_T.shape}"
        mask = attention_mask.to(hidden_states[0])
        valid_count = mask.sum() * hidden_states[0].size(-1)
        student_states = torch.stack([hidden_states[i] for i in range(len(matches))])
        teacher_states = torch.stack([hidden_states_T[j] for j in matches])
        assert student_states.shape == teacher_states.shape, f"{student_states.shape} != {teacher_states.shape}"
        if normalize_hidden:
            student_states = F.layer_norm(student_states, student_states.shape[1:])
            teacher_states = F.layer_norm(teacher_states, teacher_states.shape[1:])
        mse = F.mse_loss(student_states, teacher_states, reduction="none")
        masked_mse = (mse * mask.unsqueeze(0).unsqueeze(-1)).sum() / valid_count
        return masked_mse

class MSELossCallback(MetricCallback):
    """Callback to compute MSE loss"""

    def __init__(
        self,
        input_key: Union[str, List[str], Dict[str, str]] = None,
        output_key: Union[str, List[str], Dict[str, str]] = None,
        prefix: str = "mse_loss",
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
            multiplier (float): scale factor for the output loss.
        """
        if output_key is None:
            output_key = [
                "attention_mask",
                "input_attention_mask",
                "t_hidden_states",
                "s_hidden_states",
                "dt_hidden_states",
                "ds_hidden_states",
            ]
        super().__init__(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
            metric_fn=self.metric_fn,
            **metric_kwargs,
        )
        self._criterion = nn.MSELoss(reduction="mean")

    def metric_fn(
        self,
        attention_mask: torch.Tensor,
        input_attention_mask: torch.Tensor,
        t_hidden_states: torch.Tensor,
        s_hidden_states: torch.Tensor,
        dt_hidden_states: torch.Tensor,
        ds_hidden_states: torch.Tensor,
    ):
        """
        Computes MSE loss for given distributions
        Args:
            s_logits: tensor shape of (batch_size, seq_len, voc_size)
            t_logits: tensor shape of (batch_size, seq_len, voc_size)
            attention_mask:  tensor shape of (batch_size, seq_len, voc_size)
        Returns:
            MSE loss
        """
        #FIXME
        # sel_mask = attention_mask[:, :, None].expand_as(s_logits)
        # sel_mask = sel_mask.ge(0.5)
        # vocab_size = s_logits.size(-1)
        # s_logits_slct = torch.masked_select(s_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        # t_logits_slct = torch.masked_select(t_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        # s_logits_slct = s_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        # t_logits_slct = t_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        # mask has False at padding_idx
        normalize_hidden = True
        e_layer_ids = len(s_hidden_states)-1 #number of encoder layers in student
        d_layer_ids = len(ds_hidden_states)-1 #number of decoder layers in student
        teacher_encoder_layers = 16 # number of encoder layers in teacher 
        teacher_decoder_layers = 16 # number of decoder layers in teacher 
        e_matches = get_layers_to_supervise(
                    n_student=e_layer_ids, n_teacher=teacher_encoder_layers
                )
        d_matches = get_layers_to_supervise(
                    n_student=d_layer_ids, n_teacher=teacher_decoder_layers
                )
        hid_loss_enc = calc_hidden_loss(
                    input_attention_mask,
                    s_hidden_states,
                    t_hidden_states,
                    e_matches,
                    normalize_hidden=normalize_hidden,
                )
        hid_loss_dec = calc_hidden_loss(
                attention_mask,
                ds_hidden_states,
                dt_hidden_states,
                d_matches,
                normalize_hidden=normalize_hidden,
            )
        return hid_loss_enc + hid_loss_dec
