import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A, dims):
        if dims == 2:
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
        elif dims == 3:
            x = torch.einsum('ncvl,nvw->ncwl', (x, A))
        else:
            raise NotImplementedError('TVGCN not implemented for A of dimension ' + str(dims))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        #print("supportlen",support_len)
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a, a.dim())
            print("x1shape", x1.shape)
            out.append(x1)

            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a, a.dim())
                print("x2shape", x2.shape)
                out.append(x2)
                x1 = x2

        #s = np.array(torch.tensor(out,device='cpu')).shape
        #print("outshape", len(out))
        h = torch.cat(out, dim=1)
        print("hshape", h.shape)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class A_dynamic(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        # print('c_in=',c_in)
        super(A_dynamic, self).__init__()
        self.conv1 = nn.Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(tem_size, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        # nn.Parameter
        self.w = nn.Parameter(torch.rand(tem_size, c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)

    # seq：
    def forward(self, seq):

        c1 = seq  #b,c,n,l
        c2 = seq.permute(0, 3, 1, 2)  # b,c,n,l->b,l,c,n

        f1 = self.conv1(c1).squeeze(1)  # batch_size,num_nodes,length_time

        f2 = self.conv2(c2).squeeze(1)  # b,c,n
        #print("self.w", self.w.shape)
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2))
        #print("logits", logits.shape)

        a, _ = torch.max(logits, 1, True)
        logits = logits - a
        A_d = torch.softmax(logits, -1)
        return A_d

class tvgcn(nn.Module):
    def __init__(self, device, dropout=0.3, supports=None, num_nodes=170, ksize=20, tanhalpha=3, gcn_bool=True,
                 in_dim=3, out_dim=24, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, time_embedding="false",
                 kernel_size=5, blocks=3, layers=2, n_head=1, input_dim=4, embed_dim=24, act_function=torch.sin):
        super(tvgcn, self).__init__()
        self.num_nodes = num_nodes
        self.ksize = ksize
        self.tanhalpha = tanhalpha
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.time_embedding = time_embedding

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports
        self.n_head = n_head
        self.A_dynamic = A_dynamic(self.in_dim, self.num_nodes, self.out_dim)

        # self.supports_len += 1
        if self.time_embedding == "true":
            self.embed_dim = embed_dim // input_dim
            self.input_dim = input_dim
            self.embed_weight = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim), requires_grad=True
            )
            # torch.nn.init.xavier_uniform_(self.embed_weight)
            self.embed_bias = nn.parameter.Parameter(torch.randn(self.embed_dim), requires_grad=True)
            # torch.nn.init.uniform_(self.embed_bias, a=0.0, b=1.0)
            self.act_function = act_function

        self.idx = torch.arange(self.num_nodes)#.to(device)

        receptive_field = 1

        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)
            #print("1", len(self.supports))

        if gcn_bool:
            if self.supports is None:
                self.supports = []
            self.adpvec = nn.Parameter(torch.randn(self.out_dim, self.out_dim).to(device), requires_grad=True).to(device)
            self.supports_len += 1
            #print("2", len(self.supports))

        #print("3", len(self.supports))
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):  #layers=2
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                #print("supportslen", self.supports_len)
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1, 1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        if self.time_embedding == "true":
            # time embedding with Time2vector method
            input = input.transpose(1, 3)
            trainn = input[:, :, 0, 1:].squeeze()  # (b,t,n,f)
            x = torch.diag_embed(trainn)
            # x.shape = (bs, sequence_length, input_dim, input_dim)
            x_affine = torch.matmul(x, self.embed_weight) + self.embed_bias
            # x_affine.shape = (bs, sequence_length, input_dim, time_embed_dim)
            x_affine_0, x_affine_remain = torch.split(
                x_affine, [1, self.embed_dim - 1], dim=-1
            )
            x_affine_remain = self.act_function(x_affine_remain)
            x_output = torch.cat([x_affine_0, x_affine_remain], dim=-1)
            x_output = x_output.view(x_output.size(0), x_output.size(1), -1)
            # x_output.shape = (bs, sequence_length, input_dim * time_embed_dim)
            # # x_t = (batch, len, d_x) --> (batch, d_x, nodes, len)
            x_t = x_output.transpose(-1, 1).unsqueeze(-2).repeat(1, 1, self.num_nodes, 1)
            trainx = input.transpose(1, 3)  # (b,t,n,f)->(b,f,n,t)
            trainx = trainx[:, 0:1, :, :]
            #print("trainx.shape",trainx.shape)
            input = torch.cat((trainx, x_t), dim=1)
        print("input", input.shape)
        in_len = input.size(3)    #input:(b,f,n,t)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input   #b,feature,n,t
        print("before start", x.shape)
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.supports is not None:
            s_l = self.out_dim
            xn = input[:, :, :, -s_l:]

            adp = self.A_dynamic(xn)

            new_supports = self.supports + [torch.tensor(adp).to(self.device)]
            #print("len-new-supports", len(new_supports))

        # WaveNet layers
        for i in range(self.blocks * self.layers):  #blocks=4，layers=2
            # print(torch.cuda.memory_allocated(device='cuda:0'))

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            print("*  ", x.shape)
            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            if self.gcn_bool and self.supports is not None:
                    x = self.gconv[i](x, new_supports)
            else:
                x = self.residual_convs[i](x)
            #print("after gcn", x.shape)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        print("conv1", x.shape)
        x = self.end_conv_2(x)
        print("conv2", x.shape)
        return x





