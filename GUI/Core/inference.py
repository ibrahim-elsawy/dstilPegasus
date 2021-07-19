
from PySide2.QtCore import Signal, QObject
from transformers import AutoTokenizer, PegasusForConditionalGeneration
from torch.quantization import quantize_dynamic


modelspaths = {'Pegasus-xsum  16-16': ('D:\\Pegasus_Model\\saves\\pegasus-gigaword', 19),
               'Pegasus-xsum  SF 16-12': '',
               'Pegasus-xsum  SF 16-8': '',
               'Pegasus-xsum  SF 16-4': '',
               'Pegasus-xsum  PL 16-4': '',
               'Pegasus-xsum  PL 12-6': '',
               'Pegasus-xsum  PL 12-3': ''}

from contextlib import contextmanager
import time
import warnings

warnings.filterwarnings('ignore')


@contextmanager
def timer(msg):
    t0 = time.time()
    print(f'[{msg}] start.')
    yield
    elapsed_time = time.time() - t0
    print(f'[{msg}] done in {elapsed_time} sec.')


class InferenceClass(QObject):


    def __init__(self):
        QObject.__init__(self)

    def infer(self, text, models, quantized):
        model_ckpt = modelspaths[models.text()][0]
        Rouge_Score = modelspaths[models.text()][1]
        max_input_length = 128

        model = self.getModel(model_ckpt, max_input_length, quantized)

        tokens = self.tokenize(text=text, tokenizerName='google/pegasus-gigaword',
                               max_input_length=max_input_length)

        output_tokens, elapsed_time = self.generateSummary(model=model, tokens=tokens)

        output = self.decodeOutput(tokenizerName='google/pegasus-gigaword', tokens=output_tokens)
        return output, elapsed_time, Rouge_Score

    def getModel(self, path, max_input_length, quantized):
        with timer('Loading Model'):
            model = PegasusForConditionalGeneration.from_pretrained(path, max_length=max_input_length,
                                                                    max_position_embeddings=max_input_length)

        if quantized:
            with timer('Quantize the Model'):
                model = quantize_dynamic(model)

        return model

    def tokenize(self, text, tokenizerName, max_input_length):
        with timer('Tokenizing ...'):
            tokenizer = AutoTokenizer.from_pretrained(tokenizerName)
            token = tokenizer(text, truncation=True, padding='max_length', max_length=max_input_length,
                              return_tensors="pt")
        return token

    def generateSummary(self, model, tokens):
        with timer('Generating Summary ...'):
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            # 'set num_beams = 1' for greedy search
            t0 = time.time()
            tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=4)
            elapsed_time = time.time() - t0

        return tokens, elapsed_time

    def decodeOutput(self, tokenizerName, tokens):
        with timer('Decoding Output'):
            tokenizer = AutoTokenizer.from_pretrained(tokenizerName)
            output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
        return output
