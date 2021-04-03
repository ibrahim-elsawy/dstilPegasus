from aquvitae import dist, ST
import numpy as np
import torch    
from torch import nn
import transformers
import pandas as pd 
from transformers import AutoTokenizer
from transformers.data.data_collator import default_data_collator
from torch.utils.data import DataLoader
from typing import Iterable, Union

# import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
# import transformers
from transformers import AutoTokenizer


class LanguageModelingDataset(Dataset):
    """
    Dataset for (masked) language model task.
    Can sort sequnces for efficient padding.
    """

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: Union[
            str, transformers.models.pegasus.tokenization_pegasus_fast.PegasusTokenizerFast
        ],
        max_seq_length: int = None,
        sort: bool = True,
        lazy: bool = False,
    ):
        """
        Args:
            texts (Iterable): Iterable object with text
            tokenizer (str or tokenizer): pre trained
                huggingface tokenizer or model name
            max_seq_length (int): max sequence length to tokenize
            sort (bool): If True then sort all sequences by length
                for efficient padding
            lazy (bool): If True then tokenize and encode sequence
                in __getitem__ method
                else will tokenize in __init__ also
                if set to true sorting is unavialible
        """
        if sort and lazy:
            raise Exception(
                "lazy is set to True so we can't sort"
                " sequences by length.\n"
                "You should set sort=False and lazy=True"
                " if you want to encode text in __get_item__ function"
            )
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif isinstance(
            tokenizer, transformers.models.pegasus.tokenization_pegasus_fast.PegasusTokenizerFast
            # transformers.tokenization_utils.PreTrainedTokenizer
        ):
            self.tokenizer = tokenizer
        else:
            raise TypeError(
                "tokenizer argument should be a model name"
                + " or huggingface PreTrainedTokenizer"
            )
            # self.tokenizer = tokenizer

        self.max_seq_length = max_seq_length

        self.lazy = lazy

        if lazy:
            self.texts = texts

        if not lazy:
            pbar = tqdm(texts, desc="tokenizing texts")
            self.encoded = [
                self.tokenizer.encode(text, max_length=max_seq_length)
                for text in pbar
            ]
            if sort:
                self.encoded.sort(key=len)

        self.length = len(texts)

        self._getitem_fn = (
            self._getitem_lazy if lazy else self._getitem_encoded
        )

    def __len__(self):
        """Return length of dataloader"""
        return self.length

    def _getitem_encoded(self, idx) -> torch.Tensor:
        return torch.tensor(self.encoded[idx])

    def _getitem_lazy(self, idx) -> torch.Tensor:
        encoded = self.tokenizer.encode(
            self.texts[idx], max_length=self.max_seq_length
        )

        return torch.tensor(encoded)

    def __getitem__(self, idx):
        """Return tokenized and encoded sequence"""
        return self._getitem_fn(idx)
#****************************************************************************************
class PegasusForMLM(nn.Module):
    """
    BertForMLM

    Simplified huggingface model
    """

    def __init__(
        self,
        model_name: str = "tuner007/pegasus_paraphrase",
        output_logits: bool = True,
        output_hidden_states: bool = True,
    ):
        """
        Args:
            model_name: huggingface model name
            output_logits: same as in huggingface
            output_hidden_states: same as in huggingface
        """
        super().__init__()
        self.config = transformers.AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states,
            output_logits=output_logits
        )
        self.bert = transformers.PegasusForCausalLM.from_pretrained(
            model_name, config=self.config
        )

    def forward(self, *model_args, **model_kwargs):
        """Forward method"""
        return self.bert(*model_args, **model_kwargs)
#*********************************************************************************************************

# PATH_TO_YOUR_DATASET = "./data"
# train_df = pd.read_csv(f"{PATH_TO_YOUR_DATASET}/train.csv")
# valid_df = pd.read_csv(f"{PATH_TO_YOUR_DATASET}/valid.csv")
# valid_df = valid_df.drop('Unnamed: 0', axis=1)
# train_df = train_df.drop('Unnamed: 0', axis=1)
# train_df = train_df.dropna()
# valid_df = valid_df.dropna()

# collate_fn = default_data_collator

# tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")

# train_dataset = LanguageModelingDataset(train_df["text"].tolist(), tokenizer)
# valid_dataset = LanguageModelingDataset(valid_df["text"].tolist(), tokenizer)

train_dataset = torch.load("train_dataset1.pt")
valid_dataset = torch.load("valid_dataset1.pt")


train_dataloader = DataLoader(
    train_dataset.input_ids, batch_size=2
)
valid_dataloader = DataLoader(
    valid_dataset.input_ids, batch_size=2
)
loaders = {"train": train_dataloader, "valid": valid_dataloader}
# Load the dataset
# train_ds = torch.utils.data.DataLoader(train_df, batch_size=16)
# test_ds = torch.utils.data.DataLoader(valid_df, batch_size=16)
# model = torch.load('pg.pt')
# mol = torch.load('student_model.pt')
# Load the teacher and student model
model = torch.load('pg.pt')
mol = torch.load('st_3dec_3enc.pt')
teacher = model.bert
student = mol 
model = torch.nn.ModuleDict({"teacher": teacher, "student": student})
optimizer = torch.optim.Adam(model.parameters())

# Knowledge Distillation
student = dist(
    teacher=teacher,
    student=student,
    algo=ST(alpha=0.6, T=2.5),
    optimizer=optimizer,
    # train_ds=loaders["train"],
    train_ds=train_dataset,
    # test_ds=loaders["valid"],
    test_ds=valid_dataset,
    iterations=3000
)

torch.save(student, 'distil_student.pt')