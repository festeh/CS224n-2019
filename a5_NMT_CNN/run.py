from logging import basicConfig, INFO
from pathlib import Path

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from ignite.contrib.handlers import ProgressBar, TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.utils import apply_to_type
from torch.optim import Adam

from a5_NMT_CNN.metrics import Perplexity, Loss, BLEU
from a5_NMT_CNN.nmt_model import NMTModel
from a5_NMT_CNN.read_data import NMTDataReader


class DataIteratorWrapper:
    def __init__(self, data_iter: BasicIterator, instances, shuffle):
        self.data_iter = data_iter
        self.instances = instances
        self.shuffle = shuffle

    def __len__(self):
        return self.data_iter.get_num_batches(self.instances)

    def __iter__(self):
        return self.data_iter(self.instances, shuffle=self.shuffle, num_epochs=1)


def get_data_loader(config):
    data_reader = NMTDataReader(convert_to_lowercase=config.pop("convert_to_lowercase"))
    train_instances = data_reader.read(config.pop("train_data_path"))
    val_instances = data_reader.read(config.pop("valid_data_path"))
    vocab_path = Path(config.pop("vocab_path"))
    # firstly create vocab
    if not vocab_path.exists():
        max_vocab_size = config.pop("max_vocab_size")
        vocab = Vocabulary.from_instances(
            train_instances,
            max_vocab_size={
                "char_src": 96,
                "token_src": max_vocab_size,
                "char_trg": 96,
                "token_trg": max_vocab_size,
            },
        )
        vocab.save_to_files(vocab_path)
    else:
        vocab = Vocabulary.from_files(vocab_path)

    train_data_iter = BasicIterator(
        batch_size=config.pop("train_batch_size"), cache_instances=True
    )
    train_data_iter.index_with(vocab)

    valid_data_iter = BasicIterator(
        batch_size=config.pop("valid_batch_size"), cache_instances=True
    )
    valid_data_iter.index_with(vocab)

    return (
        vocab,
        DataIteratorWrapper(train_data_iter, train_instances, shuffle=True),
        DataIteratorWrapper(valid_data_iter, val_instances, shuffle=False),
    )


def create_nmt_trainer(model, optimizer, max_grad_norm=None, device="cpu"):
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


def create_nmt_evaluator(model: NMTModel, metrics={}, device=None, non_blocking=False):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            apply_to_type(batch, torch.Tensor, lambda x: x.to(device))
            scores = model(batch, reduce=False)
            return scores

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def run_evaluation(trainer):
    print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")


if __name__ == "__main__":
    basicConfig(level=INFO)
    config = Params.from_file("config.jsonnet")
    vocab, train_data_iter, valid_data_iter = get_data_loader(config)
    nmt_model = NMTModel(config, vocab)

    optimizer = Adam(nmt_model.parameters(), lr=config.pop("lr"))
    trainer = create_nmt_trainer(
        nmt_model,
        optimizer,
        max_grad_norm=config.pop("max_grad_norm"),
        device=nmt_model.device,
    )

    evaluator = create_nmt_evaluator(
        nmt_model,
        metrics={
            "loss": Loss(),
            "ppl": Perplexity(),
            "bleu": BLEU(vocab, valid_data_iter.instances),
        },
        device=nmt_model.device,
    )

    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    pbar = ProgressBar(persist=False, bar_format=None)
    pbar.attach(trainer, ["loss"])

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, lambda e: evaluator.run(valid_data_iter)
    )
    tb_logger = TensorboardLogger(log_dir=config.pop("results_path"))

    tb_logger.attach(
        trainer,
        log_handler=OutputHandler(tag="training", metric_names=["loss"]),
        event_name=Events.EPOCH_COMPLETED,
    )

    tb_logger.attach(
        evaluator,
        log_handler=OutputHandler(
            tag="validation", metric_names=["loss", "ppl", "bleu"], another_engine=trainer
        ),
        event_name=Events.EPOCH_COMPLETED,
    )

    trainer.run(train_data_iter, max_epochs=100)
