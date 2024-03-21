import argparse
import sys
import time
import torch
import torch.nn as nn

class args:
    name ='lstm'
    seed =int(1000 * time.time())
    dim_gram =1
    dataset ='D:\pythoncode\Paper\TravelTime0307\datasets1'
    fast =0
    screen =0
    lr =0.01
    # lr_word_vector =0.5
    epoch =1
    batch =25
    # input_size =4
    input_size =1
    hidden_size =4
    weight_decay =5e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()