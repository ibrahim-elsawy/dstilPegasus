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

#*******************************************************************************************************************

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
#**************************************************************************************************************************************
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
#**************************************************************************************************************************************
class DistilpegasusStudentModel(nn.Module):
    """
    DistilbertStudentModel

    Distil model class based on huggingface class but with
    initialization in it. Model will take vocabulary
    layers from specified teacher model
    """

    def __init__(
        self,
        teacher_model_name: str = "tuner007/pegasus_paraphrase",
        layers: List[int] = None,
        extract: bool = True,
    ):
        """
        Args:
            teacher_model_name: name of the model to distil
            layers: layers indexes to initialize
            extract: bool flag, if you want to initialize your model with
                layers of the teacher model then set this to true
        """
        super().__init__()
        if layers is None:
            layers = [0, 2, 4, 7, 9, 11, 15]
        teacher_config = transformers.AutoConfig.from_pretrained(
            teacher_model_name, output_hidden_states=True, output_logits=True
        )
        teacher = transformers.PegasusForCausalLM.from_pretrained(
            teacher_model_name, config=teacher_config
        )
        distil_sd = None
        if extract:
            distil_sd = self._extract(teacher, layers)
        if teacher_model_name == "tuner007/pegasus_paraphrase":
            f = open('a7ooo.txt', 'a')
            f.write(','.join(map(str, distil_sd)))
            f.close()
            student_config = transformers.AutoConfig.from_pretrained(
                "tuner007/pegasus_paraphrase",
                output_hidden_states=True,
                output_logits=True,
            )
            self.student = transformers.PegasusForCausalLM.from_pretrained(
                "tuner007/pegasus_paraphrase",
                config=student_config,
                state_dict=distil_sd,
            )
        elif teacher_model_name == "bert-base-cased":
            student_config = transformers.AutoConfig.from_pretrained(
                "distilbert-base-cased",
                output_hidden_states=True,
                output_logits=True,
            )
            self.student = transformers.DistilBertForMaskedLM.from_pretrained(
                "distilbert-base-cased",
                config=student_config,
                state_dict=distil_sd,
            )
        else:
            student_config = transformers.AutoConfig.from_pretrained(
                "distilbert-base-multilingual-cased",
                output_hidden_states=True,
                output_logits=True,
            )
            self.student = transformers.DistilBertForMaskedLM.from_pretrained(
                "distilbert-base-multilingual-cased",
                config=student_config,
                state_dict=distil_sd,
            )

    def forward(self, *model_args, **model_kwargs):
        """Forward nethod"""
        return self.student(*model_args, **model_kwargs)

    @staticmethod
    def _extract(
        teacher_model,
        layers: List[int],
        prefix_teacher: str = "model",
        prefix_student: str = "pegasus_student",
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts state dict from teacher model

        Args:
            teacher_model: model to extract
            layers: layers indexes to initialize
            prefix_teacher: name of the teacher model
            prefix_student: name of the student model
        """
        state_dict = teacher_model.state_dict()
        # f = open('state_dict.txt', 'a')
        # f.write(','.join(map(str, teacher_model.state_dict())))
        # f.close()
        compressed_sd = {}

        # extract embeddings
        for w in ["embed_tokens", "embed_positions"]:
            compressed_sd[
                f"{prefix_student}.decoder.{w}.weight"
            ] = state_dict[f"{prefix_teacher}.decoder.{w}.weight"]
        print('finished_11111111')
        for w in ["weight", "bias"]:
            compressed_sd[
                f"{prefix_student}.decoder.layers.0.self_attn_layer_norm.{w}"
            ] = state_dict[f"{prefix_teacher}.decoder.layers.0.self_attn_layer_norm.{w}"]
        print('finished_222222')
        # extract encoder
        std_idx = 0
        for teacher_idx in layers:
            for w in ["weight", "bias"]:
                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.attention.q_lin.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.decoder.layers.{teacher_idx}.encoder_attn.q_proj.{w}"  # noqa: E501
                ]
                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.attention.k_lin.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.decoder.layers.{teacher_idx}.encoder_attn.k_proj.{w}"  # noqa: E501
                ]
                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.attention.v_lin.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.decoder.layers.{teacher_idx}.encoder_attn.v_proj.{w}"  # noqa: E501
                ]

                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.attention.out_lin.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.decoder.layers.{teacher_idx}.self_attn.out_proj.{w}"  # noqa: E501
                ]
                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.sa_layer_norm.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.decoder.layers.{teacher_idx}.self_attn_layer_norm.{w}"  # noqa: E501
                ]

                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.ffn.lin1.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.decoder.layers.{teacher_idx}.fc1.{w}"  # noqa: E501
                ]
                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.ffn.lin2.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.decoder.layers.{teacher_idx}.fc2.{w}"  # noqa: E501
                ]
                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.output_layer_norm.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.decoder.layers.{teacher_idx}.final_layer_norm.{w}"  # noqa: E501
                ]

            std_idx += 1
        print('finished_3333333333')
        # extract vocab
        compressed_sd[f"lm_head.weight"] = state_dict[
            f"lm_head.weight"
        ]
        # compressed_sd[f"vocab_projector.bias"] = state_dict[
        #     f"cls.predictions.bias"
        # ]

        # for w in ["weight", "bias"]:
        #     compressed_sd[f"vocab_transform.{w}"] = state_dict[
        #         f"cls.predictions.transform.dense.{w}"
        #     ]
        #     compressed_sd[f"vocab_layer_norm.{w}"] = state_dict[
        #         f"cls.predictions.transform.LayerNorm.{w}"
        #     ]

        return compressed_sd
#**************************************************************************************************************************************


# train_dataset = np.load('train_dataset.npy', allow_pickle=True)
# valid_dataset = np.load('valid_dataset.npy', allow_pickle=True)
# dataset = load_dataset('csv', data_files = {'train':["./modified_train.csv"], 'valid':["./modified_valid.csv"]})
dataset = datasets.load_from_disk('gigaword')
# dataset = load_dataset("gigaword")

tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase", padding='max_length')
# dataset = datasets.load_from_disk('cnn_dataset')


# tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
# train_dataset = tokenizer(dataset['train']['document'][0:200000],padding='longest')
# valid_dataset = tokenizer(dataset['validation']['document'][0:10000],padding='longest')
# train_label = tokenizer(dataset['train']['summary'][0:200000],padding='longest')
# valid_label = tokenizer(dataset['validation']['summary'][0:10000],padding='longest')
# torch.save(train_dataset,'train_dataset1.pt')
# torch.save(valid_dataset,'valid_dataset1.pt')
# torch.save(train_label,'train_label1.pt')
# torch.save(valid_label,'valid_label1.pt')
# train_dataset = LanguageModelingDataset(dataset['train']['article'], tokenizer)
# valid_dataset = LanguageModelingDataset(dataset['validation']['article'], tokenizer)
# train_label = LanguageModelingDataset(dataset['train']['highlights'], tokenizer)
# valid_label = LanguageModelingDataset(dataset['validation']['highlights'], tokenizer)


# train_dataset = np.load("train_dataset1.npy", allow_pickle=True)
# valid_dataset = np.load("valid_dataset1.npy", allow_pickle=True)
# train_label = np.load("train_label1.npy", allow_pickle=True)
# valid_label = np.load("valid_label1.npy", allow_pickle=True)

# dataset['train']['article'] = train_dataset # dataset['validation']['article'] = valid_dataset
# dataset['train']['highlights'] = train_label
# dataset['validation']['highlights'] = valid_label

# train_dataset = torch.load("train_dataset1.pt")
# valid_dataset = torch.load("valid_dataset1.pt")
# train_label = torch.load("train_label1.pt")
# valid_label = torch.load("valid_label1.pt")
train_dataset = dataset.map(lambda e: tokenizer(e['document'][0:1000], truncation=True, padding='max_length'), batched=True)
train_label = dataset.map(lambda e: tokenizer(e['summary'][0:1000], truncation=True, padding='max_length'), batched=True)
valid_dataset = dataset.map(lambda e: tokenizer(e['document'][0:1000], truncation=True, padding='max_length'), batched=True)
valid_label = dataset.map(lambda e: tokenizer(e['summary'][0:1000], truncation=True, padding='max_length'), batched=True)
# data = {
#     'train':{'article':'', 'highlights':''},
#     'validation':{'article':'', 'highlights':''}} 
# data['train']['article'] = train_dataset
# data['validation']['article'] = valid_dataset
# data['train']['highlights'] = train_label
# data['validation']['highlights'] = valid_label
# data['train']['idx'] = dataset['train']['id']
# data['validation']['idx'] = dataset['validation']['id']

train_dataset.decode_ids = train_label.inputs_ids
valid_dataset.decode_ids = valid_label.inputs_ids

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask'], device='cuda')
valid_dataset.set_format('torch', columns=['input_ids', 'attention_mask'], device='cuda')
# dataset.save_to_disk('cnn_dataset_tensor')
train_dataloader = DataLoader(
    train_dataset['train'], batch_size=10
)
valid_dataloader = DataLoader(
    valid_dataset['validation'], batch_size=10
)
loaders = {"train": train_dataloader, "valid": valid_dataloader}



teacher = torch.load('pg.pt')
student = torch.load('st_3dec_3enc.pt')

model = torch.nn.ModuleDict({"teacher": teacher, "student": student})

callbacks = {
    "masked_lm_loss": MaskedLanguageModelCallback(),
    "mse_loss": MSELossCallback(),
    "cosine_loss": CosineLossCallback(),
    "kl_div_loss": KLDivLossCallback(),
    "loss": MetricAggregationCallback(
        prefix="loss",
        mode="weighted_sum",
        metrics={
            "cosine_loss": 1.0,
            "masked_lm_loss": 1.0,
            "kl_div_loss": 1.0,
            "mse_loss": 1.0
        }
    ),
    "optimizer": dl.OptimizerCallback(),
    "perplexity": PerplexityMetricCallbackDistillation()
}


runner = DistilMLMRunner(device=torch.device("cuda"))
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
runner.train(
    model=model,
    optimizer=optimizer,
    loaders=loaders,
    verbose=True,
    check=True,
    callbacks=callbacks,
)