from allennlp.common import Params
from torch.nn import Module, Conv1d


class CNN(Module):
    def __init__(self, config:Params):
        super().__init__()
        self.kernel_size = 5
        self.n_filters = config.pop("n_filters")

        self.conv = Conv1d(out_channels=self.n_filters, kernel_size=self.kernel_size)

    def forward(self, x_reshaped):
        x_conv = self.conv(x_reshaped)