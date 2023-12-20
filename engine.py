import torch.optim as optim
from model import *
import util
class trainer():
    def __init__(self, scaler, in_dim, seq_length, nhid, dropout, lrate, wdecay, num_nodes, ksize, tanhalpha, device,
                 supports, gcn_bool, time_embedding_bool):
        self.model = tvgcn(device, dropout, supports=supports, num_nodes=num_nodes, ksize=ksize, tanhalpha=tanhalpha, gcn_bool=gcn_bool,
                           in_dim=in_dim, out_dim=seq_length, residual_channels=nhid,
                           dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16, time_embedding=time_embedding_bool)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):

        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)  #input (b,f,n,t)
        output = output.transpose(1, 3)
        # output = [batch_size, 12, num_nodes, 1]
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        # input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse
