from pathlib import Path

from allennlp.common import Params
from allennlp.data import Vocabulary, DataIterator
from allennlp.data.iterators import BasicIterator

from a5_NMT_CNN.nmt_model import NMTModel
from a5_NMT_CNN.read_data import NMTDataReader

if __name__ == '__main__':
    config = Params.from_file("config.jsonnet")
    vocab_path = Path(config.pop("vocab_path"))
    data_reader = NMTDataReader(convert_to_lowercase=config.pop("convert_to_lowercase"))
    instances = data_reader.read(config.pop("train_data_path"))
    # firstly create vocab
    if not vocab_path.exists():
        max_vocab_size = config.pop("max_vocab_size")
        vocab = Vocabulary.from_instances(
            instances,
            max_vocab_size={"char_src": 96, "token_src": max_vocab_size,
                            "char_trg": 96, "token_trg": max_vocab_size})
        vocab.save_to_files(vocab_path)
    else:
        vocab = Vocabulary.from_files(vocab_path)

    data_iter = BasicIterator(batch_size=config.pop("batch_size"), cache_instances=True)
    data_iter.index_with(vocab)
    sample = next(iter(data_iter(instances, shuffle=False)))
    nmt_model = NMTModel(config, vocab)
    nmt_model(sample)
