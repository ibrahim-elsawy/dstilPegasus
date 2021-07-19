# Loss Callbacks
*These are all loss functions used in our project in our four distillation methods.*

1. ### **Pseudo Labeling method**
    *The used loss functions in this method is*
    - [cross entropy loss](https://github.com/ibrahim-elsawy/dstilPegasus/blob/main/src/callbacks/Cross_ent_loss.py)
    ![Pseudo labeling training loop](https://github.com/ibrahim-elsawy/dstilPegasus/blob/main/src/callbacks/images/pl.png)
2. ### **Knowledge Distillation method**
    *The used loss functions in this method are*
    - [cross entropy loss](https://github.com/ibrahim-elsawy/dstilPegasus/blob/main/src/callbacks/Cross_ent_loss.py)
    - [label smoothed loss](https://github.com/ibrahim-elsawy/dstilPegasus/blob/main/src/callbacks/label_smoothed_callback.py)
    - [Mean Square Error loss](https://github.com/ibrahim-elsawy/dstilPegasus/blob/main/src/callbacks/mse_loss_callback.py)
    - [Kullback-Leibler kl loss](https://github.com/ibrahim-elsawy/dstilPegasus/blob/main/src/callbacks/KL_loss_callback.py)
    ![Knowledge Distillation training loop](https://github.com/ibrahim-elsawy/dstilPegasus/blob/main/src/callbacks/images/kd.png)
3. ### **Shrink and finetune method**
    *The used loss functions in this method is*
    - [cross entropy loss](https://github.com/ibrahim-elsawy/dstilPegasus/blob/main/src/callbacks/Cross_ent_loss.py)
    ![Shrink and finetune training loop](https://github.com/ibrahim-elsawy/dstilPegasus/blob/main/src/callbacks/images/sft.png)
4. ### **Teacher assistant TA method**
    *The used loss functions in this method is*
    - [cross entropy loss](https://github.com/ibrahim-elsawy/dstilPegasus/blob/main/src/callbacks/Cross_ent_loss.py)
    ![TA training loop](https://github.com/ibrahim-elsawy/dstilPegasus/blob/main/src/callbacks/images/TA.png)


