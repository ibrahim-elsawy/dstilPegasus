import torch
import transformers
from catalyst import dl
from src.runners import DistilMLMRunner
# from src.models import DistilpegasusStudentModel, PegasusForMLM #BertForMLM  DistilbertStudentModel
from catalyst.core import MetricAggregationCallback
from torch.utils.data import DataLoader
from src.callbacks import (
    CosineLossCallback,
    KLDivLossCallback,
    MaskedLanguageModelCallback,
    MSELossCallback,
    PerplexityMetricCallbackDistillation,
    SmoothingLossCallback
)
import pandas as pd
from typing import Iterable, Union

# import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
# import transformers
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling
import numpy as np
from transformers.data.data_collator import default_data_collator
from torch import nn
from typing import Dict, List
from datasets import load_dataset
import datasets
from datasets import Dataset


# train_dataset = datasets.load_from_disk("./xsumtrainDataset_1")
# valid_dataset = datasets.load_from_disk("./xsumvalidDataset_1")
train_dataset = datasets.load_from_disk('./trainDataset')
valid_dataset = datasets.load_from_disk('./validDataset')
teacher = torch.load('teacher_model.pt', map_location=torch.device('cuda')).to('cuda')
student = torch.load('st_3dec_3enc_2.pt', map_location=torch.device('cuda')).to('cuda')
# student = torch.load('trained_student.pt', map_location=torch.device('cuda')).to('cuda')
########################### freezing the emb layers:
for param in student.model.shared.parameters():
  param.requires_grad = False
for param in student.model.encoder.embed_tokens.parameters():
  param.requires_grad = False
for param in student.model.encoder.embed_positions.parameters():
  param.requires_grad = False
for param in student.model.decoder.embed_tokens.parameters():
  param.requires_grad = False
for param in student.model.decoder.embed_positions.parameters():
  param.requires_grad = False
  
model = torch.nn.ModuleDict({"teacher": teacher, "student": student})





# train_data = Dataset.from_dict(train_dataset)
# valid_data = Dataset.from_dict(valid_dataset)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask'], device='cuda')
valid_dataset .set_format('torch', columns=['input_ids', 'attention_mask'], device='cuda')




train_dataloader = DataLoader(
    #train_dataset['train']
    train_dataset, batch_size=16
)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=16
)
loaders = {"train": train_dataloader, "valid": valid_dataloader}


callbacks = {
    # "masked_lm_loss": MaskedLanguageModelCallback(),
    "mse_loss": MSELossCallback(),
    "cosine_loss": CosineLossCallback(),
    "kl_div_loss": KLDivLossCallback(),
    "label_smooth_loss": SmoothingLossCallback(),
    "loss": MetricAggregationCallback(
        prefix="loss",
        mode="weighted_sum",
        metrics={
            "cosine_loss": 1.0,
            # "masked_lm_loss": 1.0,
            "kl_div_loss": 1.0,
            "mse_loss": 1.0,
            "label_smooth_loss": 1.0,
        }
    ),
    "optimizer": dl.OptimizerCallback(),
    # "perplexity": PerplexityMetricCallbackDistillation()
}



runner = DistilMLMRunner(device=torch.device("cuda"))
runner.device = torch.device("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
runner.train(
    model=model,
    optimizer=optimizer, 
    loaders=loaders,
    verbose=True,
    #check=True,
    callbacks=callbacks,
    num_epochs = 3,
    #check = True,
    # engine='cpu',
)