from typing import Dict
from collections import OrderedDict

from catalyst import dl
from catalyst.dl.utils import check_ddp_wrapped
import torch
import gc 


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id = 0, decoder_start_token_id = 0):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
    
class DistilMLMRunnerFT(dl.Runner):
    """Simplified huggingface Distiller wrapped with catalyst"""
    
    def _handle_batch(self, batch: Dict[str, torch.Tensor]):
        if check_ddp_wrapped(self.model):
             student = (
                #self.model.module["teacher"],
                self.model.module["student"],
            )
        else:
            student = self.model["student"]
        if(self.epoch == 2 and self.loader_batch_step == 1 and self.stage_name == "train"):
                torch.save({ 
                        'epoch': self.epoch,
                        'model_state_dict': student.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.epoch_metrics,
                }, 'trained_student2.pt')
             
        student.to('cuda')
        batch["input_ids"] = batch["input_ids"].to('cuda')
        batch["attention_mask"] = batch["attention_mask"].to('cuda')
        batch['decode_ids'] = shift_tokens_right(batch['decode_ids'], 0)
        batch['decode_ids'] = batch['decode_ids'].to('cuda')        
        studentOutput =student( torch.cuda.LongTensor( batch["input_ids"]), batch["attention_mask"], batch['decode_ids'], output_attentions=True, output_hidden_states=True)
        s_logits = studentOutput.logits
        s_hidden_states = studentOutput.encoder_hidden_states
        ds_hidden_states = studentOutput.decoder_hidden_states
        gc.collect()
        torch.cuda.empty_cache()
        self.output = OrderedDict()
        self.output["attention_mask"] = shift_tokens_right(batch['decode_mask'], 0)
        self.output["input_attention_mask"] = batch["attention_mask"]
        self.output["s_hidden_states"] = s_hidden_states



        self.output["ds_hidden_states"] = ds_hidden_states


        self.output["s_logits"] = s_logits
        self.output["target"] = batch["decode_ids"]
        del student
        del batch
        del studentOutput
        torch.cuda.empty_cache()
