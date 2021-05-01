import torch
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
para = torch.load('para2.pt')
para = torch.cuda.LongTensor(para)
input_shape = para.size()
para = para.view(-1, input_shape[-1])
model = torch.load('teacher_model.pt', map_location=torch.device('cuda'))
translated = model.generate(para , max_length=60, num_return_sequences=1, num_beams=1)
print(translated)