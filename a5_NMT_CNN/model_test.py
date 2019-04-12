import pickle

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import TokenCharactersIndexer

from a5_NMT_CNN.character_tokenizer import MyCharacterTokenizer
from a5_NMT_CNN.nmt_model import NMTModel
from a5_NMT_CNN.read_data import NMTDataReader

config = Params.from_file("config.jsonnet")
config["device"] = "cpu"
with open(config.pop("test_instances_path"), "rb") as f:
    instances = pickle.load(f)
vocab = Vocabulary.from_files(config.pop("vocab_path"))
model = NMTModel(config, vocab).eval()
model.load_state_dict(torch.load("my_model/nmt_mymodel_4.pth"))
test_iterator = BasicIterator(batch_size=1)
test_iterator.index_with(vocab)
for idx, test_data_sample in enumerate(test_iterator(instances, shuffle=False)):
    # noinspection PyTypeChecker
    with torch.no_grad():
        hyps = model.beam_search(
            test_data_sample,
            TokenCharactersIndexer(
                "char_trg",
                min_padding_length=5,
                character_tokenizer=MyCharacterTokenizer(max_length=21),
            )
        )
    print(hyps[0].value,
          "->",
          " ".join([str(t) for t in instances[idx]["target_sentence"].tokens[1:-1]]))
    if idx == 10:
        break
