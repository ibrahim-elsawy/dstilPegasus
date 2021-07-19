# Runners
We have two implementations of runner in two files which are [runner.py](https://github.com/ibrahim-elsawy/dstilPegasus/blob/main/src/runners/runners.py) and [runner_finetune.py](https://github.com/ibrahim-elsawy/dstilPegasus/blob/main/src/runners/runner_finetune.py) in both of them we create class called 
"DistilMLMRunnerFT" which inherit from "dl.runner" class and we implement "_handle_batch" method which acts as a callback after each batch finished in training or validation loop it will be called, in this method we feedforward our student model with batch and prepare the output of the feedforward of the model to be passed to [loss functions callbacks](https://github.com/ibrahim-elsawy/dstilPegasus/tree/main/src/callbacks) 
 
