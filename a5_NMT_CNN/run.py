from logging import basicConfig, INFO
from pathlib import Path

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary, DataIterator
from allennlp.data.iterators import BasicIterator
from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_trainer, Engine, Events
from ignite.utils import convert_tensor, apply_to_type
from torch.optim import Adam

from a5_NMT_CNN.nmt_model import NMTModel
from a5_NMT_CNN.read_data import NMTDataReader


def get_data_loader(config):
    data_reader = NMTDataReader(convert_to_lowercase=config.pop("convert_to_lowercase"))
    instances = data_reader.read(config.pop("train_data_path"))
    vocab_path = Path(config.pop("vocab_path"))
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
    return vocab, data_iter(instances, shuffle=False)


def create_nmt_trainer(model, optimizer, max_grad_norm=None, device="cpu", ):
    model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        apply_to_type(batch, torch.Tensor, lambda x: x.to(device))
        scores = model(batch)
        loss = -scores.mean()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


def log_training_loss(trainer):
    print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")


if __name__ == '__main__':
    basicConfig(level=INFO)
    config = Params.from_file("config.jsonnet")
    vocab, train_data_iter = get_data_loader(config)
    nmt_model = NMTModel(config, vocab)

    optimizer = Adam(nmt_model.parameters(), lr=config.pop("lr"))
    trainer = create_nmt_trainer(nmt_model, optimizer, max_grad_norm=config.pop("max_grad_norm"))

    pbar = ProgressBar(persist=True)
    # pbar.attach(trainer, ['loss'])

    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)

    trainer.run(train_data_iter, max_epochs=100)
