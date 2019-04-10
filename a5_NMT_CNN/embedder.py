from allennlp.common import Params
from einops import rearrange
from torch.nn import Module, Embedding

from a5_NMT_CNN.cnn import CNN
from a5_NMT_CNN.highway import HighwayNetwork


class NMTEmbedder(Module):
    def __init__(self, vocab_size, params: Params):
        super().__init__()
        self.vocab_size = vocab_size
        self.char_emb_size = params.get("char_emb_size")
        self.dropout_rate = params.get("dropout_rate")

        self.init_char_embs = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.char_emb_size,
            padding_idx=0,
        )
        self.cnn = CNN(params)
        self.highway_network = HighwayNetwork(
            Params(
                {"dropout_rate": self.dropout_rate, "input_size": self.char_emb_size}
            )
        )

    def forward(self, input_):
        embedded_input = self.init_char_embs(input_)
        # bs: batch_size,
        # mw: max words in sentence,
        # ml: max word length,
        # e: emb dim
        reshaped_input = rearrange(
            embedded_input, "bs mw ml e -> (bs mw) e ml"
        )
        pooled_embs = self.cnn(reshaped_input)
        word_embs = self.highway_network(pooled_embs)
        reshaped_word_embs = rearrange(
            word_embs, "(bs mw) e -> bs mw e", bs=input_.size(0)
        )
        return reshaped_word_embs
