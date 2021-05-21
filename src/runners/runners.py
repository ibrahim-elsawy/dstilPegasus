from typing import Dict
from collections import OrderedDict

from catalyst import dl
from catalyst.dl.utils import check_ddp_wrapped
import torch
import gc 


# def shift_tokens_right(input_ids, pad_token_id):
#         """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
#         prev_output_tokens = input_ids.clone()
#         #print(pad_token_id, input_ids)
#         x= (input_ids.ne(pad_token_id).sum(dim=1) - 1)
#         index_of_eos = x.unsqueeze(-1)
#         prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
#         prev_output_tokens[:, 1:] = input_ids[:, :-1]
#         return prev_output_tokens
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
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
        if(self.epoch == 2 and self.loader_batch_step == 1 and self.stage_name == "train"):
                torch.save({ 
                        'epoch': self.epoch,
                        'model_state_dict': student.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.epoch_metrics,
                }, 'trained_student2.pt')
                
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
#         decoder_ids = torch.zeros(*(batch["input_ids"].shape[0], 60),device='cuda', dtype=torch.long)
        batch['decode_ids'] = shift_tokens_right(batch['decode_ids'], 0)
        batch['decode_ids'] = batch['decode_ids'].to('cuda')
        with torch.no_grad():
            teacherOutput =teacher( torch.cuda.LongTensor( batch["input_ids"]), batch["attention_mask"], batch['decode_ids'], output_attentions=True, output_hidden_states=True)
            t_logits = teacherOutput.logits
            t_hidden_states = teacherOutput.encoder_hidden_states 
#             t_attention_mask = teacherOutput.decoder_attentions
        
        studentOutput =student( torch.cuda.LongTensor( batch["input_ids"]), batch["attention_mask"], batch['decode_ids'], output_attentions=True, output_hidden_states=True)
        s_logits = studentOutput.logits
        s_hidden_states = studentOutput.encoder_hidden_states
        dt_hidden_states = teacherOutput.decoder_hidden_states
        ds_hidden_states = studentOutput.decoder_hidden_states
#         s_attention_mask = studentOutput.decoder_attentions
        del teacherOutput
        gc.collect()
        torch.cuda.empty_cache()
        ###################### to use the model outputs
        #max_length = batch["decode_ids"].shape[-1]
        #genLabel = teacher.generate(batch["input_ids"], attention_mask=batch["attention_mask"], max_length=max_length, num_beams=1, num_return_sequences=1)
        #size = genLabel.shape
        #genLabel = torch.cat((genLabel, torch.cuda.LongTensor(size=(size[0], max_length - size[1])).fill_(0)), dim=1)
        ###########################
        self.output = OrderedDict()
        self.output["attention_mask"] = shift_tokens_right(batch['decode_mask'], 0)
        self.output["input_attention_mask"] = batch["attention_mask"]
        self.output["t_hidden_states"] = t_hidden_states
        self.output["s_hidden_states"] = s_hidden_states



        #FIXME
        self.output["dt_hidden_states"] = dt_hidden_states
        self.output["ds_hidden_states"] = ds_hidden_states


        self.output["s_logits"] = s_logits
        self.output["t_logits"] = t_logits
        self.output["target"] = batch["decode_ids"]
        #self.output["target"] = genLabel
        #del genLabel
        #FIXME
        teacher.to('cpu')
        #student.to('cpu')
        torch.cuda.empty_cache()
