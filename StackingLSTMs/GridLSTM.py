import torch
import torch.nn as nn
import torch.nn.functional as F
from args import args

class GridLSTMCell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(GridLSTMCell,self).__init__()
        
        self.input_gate = nn.Sequential(
            #(batch_size, hidden_size)
            nn.Linear(input_size+hidden_size*2, hidden_size),
            # (batch_size, input_size+hidden_size)
            nn.Sigmoid()
        )
        
        self.forget_gate1 = nn.Sequential(
            #(batch_size, hidden_size)
            nn.Linear(input_size+hidden_size*2, hidden_size),
            # (batch_size, input_size+hidden_size)
            nn.Sigmoid()
        )      

        self.forget_gate2 = nn.Sequential(
            #(batch_size, hidden_size)
            nn.Linear(input_size+hidden_size*2, hidden_size),
            # (batch_size, input_size+hidden_size)
            nn.Sigmoid()
        )   

        self.output_gate = nn.Sequential(
            #(batch_size, hidden_size)
            nn.Linear(input_size+hidden_size*2, hidden_size),
            # (batch_size, input_size+hidden_size)
            nn.Sigmoid()
        )

        self.c_hat = nn.Sequential(
            #(batch_size, hidden_size)
            nn.Linear(input_size+hidden_size*2, hidden_size),
            # (batch_size, input_size+hidden_size)
            nn.Tanh()
        )
        
        
    def forward(self, x_t, hc_t_1, hc_t_2):
        #x_t.shape = (batch_size,input_size)
        #h_t_1.shape = c_t_1.shape=(batch_size,hidden_size)
        h_t_1, c_t_1 = hc_t_1
        h_t_2, c_t_2 = hc_t_2
        
        #xh.shape=(batch_size,input_size+hidden_size)
        xh = torch.cat([x_t,h_t_1,h_t_2], dim=1)
        i_t = self.input_gate(xh)
        f_t1 = self.forget_gate1(xh)
        f_t2 = self.forget_gate1(xh)
        o_t = self.output_gate(xh)
        c_hat = self.c_hat(xh)
        
        c_t = f_t1*c_t_1 + f_t2*c_t_2 +i_t*c_hat
        h_t = o_t*torch.tanh(c_t)
        return h_t,c_t

class GridLSTM(nn.Module):
    #x (batch_size,seq_len,input_size)
    #h (batch_size,seq_len,hidden_size)
    def __init__(self):
        super(GridLSTM,self).__init__()
        self.cell = GridLSTMCell(args.input_size, args.hidden_size)
        self.linear = nn.Linear(args.hidden_size*7*4, 1)
        # self.linear2 = nn.Linear(args.hidden_size, 4)
        
    def forward(self, x, hc_0=None, hc2_list=None): #(h_0,c_0)
        #x.shape=(batch_size,seq_len, input_size)
        batch_size, seq_len, input_size = x.size()
        hidden_size = args.hidden_size
        
        cell= self.cell
        if hc_0 is not None:
            h_0,c_0=hc_0
        else:
            h_0 = torch.zeros(batch_size,hidden_size).to(args.device)
            c_0 = torch.zeros(batch_size,hidden_size).to(args.device)
        
        if hc2_list is not None:
            h2_list, c2_list = hc2_list
            h2_list = h2_list.unsqueeze(0)
            c2_list = c2_list.unsqueeze(0)
        else:
            h2_list = torch.empty(0).to(args.device)
            c2_list = torch.empty(0).to(args.device)
            h2_0 = torch.zeros(batch_size,hidden_size).to(args.device)
            c2_0 = torch.zeros(batch_size,hidden_size).to(args.device)
            for t in range(seq_len):
                h2_list = torch.cat([h2_list,h2_0],dim=0)
                c2_list = torch.cat([c2_list,c2_0],dim=0)
            h2_list = h2_list.unsqueeze(0)
            c2_list = c2_list.unsqueeze(0)
        h_t_1=h_0
        c_t_1=c_0
        h_list = torch.empty(0).to(args.device)
        c_list = torch.empty(0).to(args.device)
        # Repetitively using LSTMCell to process input data
        for t in range(seq_len):
            #x_t.shape = (batch_size ,1, input_size)
            x_t=x[:,t,:]
            h2_t_1=h2_list[:,t,:]
            c2_t_1=c2_list[:,t,:]
            h_t ,c_t=cell(x_t,(h_t_1,c_t_1),(h2_t_1,c2_t_1))
            
            h_list = torch.cat([h_list,h_t],dim=0)
            c_list = torch.cat([c_list,c_t],dim=0)
            h_t_1, c_t_1 = h_t, c_t
        
        #h_list = [(batch_size,hidden_size)*seq_len]
        # [(batch_size,1,hidden_size)]
        #h.shape=(batch_size,seq_len,hidden_size)
        
        # h = torch.stack(h_list,dim=1)
        return h_list, c_list