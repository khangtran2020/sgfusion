import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Variable


class LSTMTarget(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        hidden_dim: int = 64,
        context_final_dim: int = 32,
    ):
        super(LSTMTarget, self).__init__()

        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.context_final_dim = context_final_dim
        total_input_dim = self.input_dim
        self.context1_dim = self.input_dim + 1  # 4
        self.context2_dim = 1  # 1
        self.context_layer_1 = nn.LSTM(
            input_size=self.context1_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.context_layer_2 = nn.LSTM(
            input_size=self.context2_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.dropout_context = nn.Dropout(self.dropout)
        self.project = nn.Linear(self.hidden_dim * 2, self.context_final_dim)
        total_input_dim += self.context_final_dim
        self.lstm_stacked = nn.LSTM(
            input_size=total_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.lstm_stacked = self.lstm_stacked
        self.linear = self.linear

    def init_hidden(self, x, n_layers=1):
        return Variable(torch.zeros(n_layers, x.size(0), self.hidden_dim))

    def embed_inputs(self, batch, device):
        inputs = Variable(torch.tensor(batch[0]["input"])).to(device)
        context_input_1 = batch[0]["context_input_1"]
        context_input_2 = batch[0]["context_input_2"]
        context_input_1 = Variable(torch.from_numpy(context_input_1).float()).to(device)
        context_input_2 = Variable(torch.from_numpy(context_input_2).float()).to(device)
        context_input_1 = self.dropout_context(context_input_1)  # [bs, 450, 4]
        context_input_2 = self.dropout_context(context_input_2)  # [bs, 450, 1]
        hidden_1 = self.init_hidden(inputs).to(device)  # [1, bs, 64]
        cell_1 = self.init_hidden(inputs).to(device)  # [1, bs, 64]
        hidden_2 = self.init_hidden(inputs).to(device)  # [1, bs, 64]
        cell_2 = self.init_hidden(inputs).to(device)  # [1, bs, 64]
        out1, (_, _) = self.context_layer_1(
            context_input_1, (hidden_1, cell_1)
        )  # [bs, 450, 64]
        out2, (_, _) = self.context_layer_2(
            context_input_2, (hidden_2, cell_2)
        )  # [bs, 450, 64]
        out12 = torch.cat([out1, out2], dim=-1)  # [bs, 450, 128]
        all_outputs = self.project(out12)
        all_inputs = torch.cat([inputs, all_outputs], dim=-1)
        outputs = torch.tensor(batch[1]).float()
        return all_inputs, outputs

    def forward(
        self,
        embedded_inputs: Tensor,
    ) -> Tensor:
        h_t = self.init_hidden(
            embedded_inputs, n_layers=2
        ).float()  # [num_layers(2), batch, hid(64)]
        c_t = self.init_hidden(embedded_inputs, n_layers=2).float()
        result, (h_t, c_t) = self.lstm_stacked(
            embedded_inputs, (h_t, c_t)
        )  # result.shape = [bs, 450, hid]
        result = F.selu(self.linear(result))  # [bs, 450, 1]
        return result
