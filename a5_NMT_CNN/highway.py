from allennlp.common import Params
from torch import sigmoid
from torch.nn import Module, Linear, Dropout
from torch.nn.functional import relu


class HighwayNetwork(Module):
    def __init__(self, config: Params):
        super().__init__()
        self.input_size = config.pop("input_size")
        self.dropout_rate = config.pop("dropout_rate")

        # TODO: tweak biases
        self.proj_hidden = Linear(self.input_size, self.input_size, bias=False)
        self.proj_gate = Linear(self.input_size, self.input_size)
        self.dropout = Dropout(self.dropout_rate)

    def forward(self, x_conv_out):
        x_proj = relu(self.proj_hidden(x_conv_out))
        x_gate = sigmoid(self.proj_gate(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        return self.dropout(x_highway)


