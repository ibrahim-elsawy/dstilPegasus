from transformers import AutoTokenizer
import torch
# from datasets import load_dataset
import datasets

dataset = datasets.load_from_disk('cnn_dataset')

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
# torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_dataset = tokenizer(dataset['train']['article']).to(torch_device)
# valid_dataset = tokenizer(dataset['validation']['article'],padding='longest')
# train_label = tokenizer(dataset['train']['highlights'],padding='longest')
# valid_label = tokenizer(dataset['validation']['highlights'],padding='longest')
torch.save(train_dataset,'train_dataset1.pt')
# torch.save(valid_dataset,'valid_dataset1.pt')
# torch.save(train_label,'train_label1.pt')
# torch.save(valid_label,'valid_label1.pt')