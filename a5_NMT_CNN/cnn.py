from allennlp.common import Params
from torch.nn import Module, Conv1d, MaxPool1d
from torch.nn.functional import max_pool1d


class CNN(Module):
    def __init__(self, config: Params):
        super().__init__()
        self.kernel_size = 5
        self.char_emb_size = config.pop("char_emb_size")
        self.n_filters = config.pop("n_filters")

        # TODO: tweak bias, intuitively should be False
        self.conv = Conv1d(
            in_channels=self.char_emb_size,
            out_channels=self.n_filters,
            kernel_size=self.kernel_size,
            bias=False
        )

    def forward(self, x_reshaped):
        x_conv = self.conv(x_reshaped)
        return max_pool1d(x_conv, kernel_size=x_conv.size(2)).squeeze(2)
