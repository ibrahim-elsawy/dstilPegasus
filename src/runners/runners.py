from typing import Dict
from collections import OrderedDict

from catalyst import dl
from catalyst.dl.utils import check_ddp_wrapped
import torch
import gc 

class DistilMLMRunner(dl.Runner):
    """Simplified huggingface Distiller wrapped with catalyst"""

    def _handle_batch(self, batch: Dict[str, torch.Tensor]):
        if check_ddp_wrapped(self.model):
            teacher, student = (
                self.model.module["teacher"],
                self.model.module["student"],
            )
        else:
            teacher, student = self.model["teacher"], self.model["student"]

        teacher.to('cuda')
        student.to('cuda')
        teacher.eval()  # manually set teacher model to eval mode
        batch["input_ids"] = batch["input_ids"].to('cuda')
        batch["attention_mask"] = batch["attention_mask"].to('cuda')
        # decoder_mask = decoder_inputs != 0
        # decoder_mask = torch.Tensor(batch['decode_mask'])
        # decoder_ids = torch.Tensor(batch['decode_ids'])
        # listOfIds = map(lambda x : torch.cuda.LongTensor(x), batch['decode_ids'])
        # listOfMasks = map(lambda x : torch.cuda.LongTensor(x), batch['decode_mask'])
        # decoder_mask = torch.stack(list(listOfMasks))
        # decoder_ids = torch.stack(list(listOfIds))
        decoder_ids = torch.zeros(*(batch["input_ids"].shape[0], 60),device='cuda', dtype=torch.long)
        with torch.no_grad():
            teacherOutput =teacher( torch.cuda.LongTensor( batch["input_ids"]), batch["attention_mask"], decoder_ids, output_attentions=True)
            t_logits = teacherOutput.logits
            t_hidden_states = teacherOutput.encoder_last_hidden_state 
            t_attention_mask = teacherOutput.decoder_attentions
        
        studentOutput =student( torch.cuda.LongTensor( batch["input_ids"]), batch["attention_mask"], decoder_ids, output_attentions=True)
        s_logits = studentOutput.logits
        s_hidden_states = studentOutput.encoder_last_hidden_state
        s_attention_mask = studentOutput.decoder_attentions
        gc.collect()
        torch.cuda.empty_cache()
        self.output = OrderedDict()
        self.output["attention_mask"] = s_attention_mask
        self.output["t_hidden_states"] = t_hidden_states
        self.output["s_hidden_states"] = s_hidden_states
        self.output["s_logits"] = s_logits
        self.output["t_logits"] = t_logits
        #FIXME
        teacher.to('cpu')
        # student.to('cpu')
        torch.cuda.empty_cache()
