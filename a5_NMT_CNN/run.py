import pickle
from functools import partial
from logging import basicConfig, INFO, info
from pathlib import Path

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from ignite.contrib.handlers import ProgressBar, TensorboardLogger, LRScheduler
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage
from ignite.utils import apply_to_type
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from a5_NMT_CNN.metrics import Perplexity, Loss, BLEU
from a5_NMT_CNN.nmt_model import NMTModel
from a5_NMT_CNN.read_data import NMTDataReader


class DataIteratorWrapper:
    def __init__(self, vocab: Vocabulary, instances, batch_size, shuffle):
        self.data_iter = BasicIterator(
            batch_size=batch_size, cache_instances=True
        )
        self.data_iter.index_with(vocab)
        self.instances = instances
        self.shuffle = shuffle

    def __len__(self):
        return self.data_iter.get_num_batches(self.instances)

    def __iter__(self):
        return self.data_iter(self.instances, shuffle=self.shuffle, num_epochs=1)


def get_data_loader(config):
    data_reader = NMTDataReader(convert_to_lowercase=config.pop("convert_to_lowercase"))
    train_instances_path = Path(config.pop("train_instances_path"))
    valid_instances_path = Path(config.pop("valid_instances_path"))
    create_vocab_s_nulya = False
    if Path(train_instances_path).exists():
        info("Loading tokenized instances")
        with train_instances_path.open("rb") as f:
            train_instances = pickle.load(f)
        with valid_instances_path.open("rb") as f:
            valid_instances = pickle.load(f)
    else:
        info("Tokenizing instances...")
        create_vocab_s_nulya = True
        train_instances = data_reader.read(config.pop("train_data_path"))
        valid_instances = data_reader.read(config.pop("valid_data_path"))
        train_instances_path.parent.mkdir(parents=True)
        with train_instances_path.open("wb") as f:
            pickle.dump(train_instances, f, protocol=pickle.HIGHEST_PROTOCOL)
        with valid_instances_path.open("wb") as f:
            pickle.dump(valid_instances, f, protocol=pickle.HIGHEST_PROTOCOL)
    vocab_path = Path(config.pop("vocab_path"))
    if create_vocab_s_nulya or not vocab_path.exists():
        max_vocab_size = config.pop("max_vocab_size")
        max_characters = config.pop("max_characters")
        vocab = Vocabulary.from_instances(
            train_instances,
            max_vocab_size={
                "char_src": max_characters,
                "token_src": max_vocab_size,
                "char_trg": max_characters,
                "token_trg": max_vocab_size,
            },
        )
        vocab.save_to_files(vocab_path)
    else:
        vocab = Vocabulary.from_files(vocab_path)

    return (
        vocab,
        DataIteratorWrapper(vocab, train_instances, shuffle=True, batch_size=config.pop("train_batch_size")),
        DataIteratorWrapper(vocab, valid_instances, shuffle=False, batch_size=config.pop("valid_batch_size")),
    )


def create_nmt_trainer(model, optimizer, max_grad_norm=None, device=None):
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        batch = apply_to_type(batch, torch.Tensor, lambda x: x.to(device))
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
            batch = apply_to_type(batch, torch.Tensor, lambda x: x.to(device))
            scores = model(batch, reduce=False)
            return scores

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def reduce_on_plateau(trainer, scheduler):
    scheduler.step(trainer.state.metrics['loss'])
    trainer.state.metrics['lr'] = scheduler.optimizer.param_groups[0]['lr']


def run_evaluation(trainer):
    epoch = trainer.state.epoch
    print(f"Epoch[{epoch}] Loss: {trainer.state.output:.2f}")


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


    def get_best(engine):
        return engine.state.metrics['bleu']


    saver = ModelCheckpoint(
        config.pop("model_path"),
        "nmt",
        n_saved=1, create_dir=True, score_function=get_best, require_empty=False)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, saver, {'mymodel': nmt_model})

    scheduler = ReduceLROnPlateau(
        optimizer, factor=config.pop("lr_decay_factor"), patience=config.pop("patience")
    )

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, partial(reduce_on_plateau, scheduler=scheduler)
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
            tag="validation",
            metric_names=["loss", "ppl", "bleu", 'lr'],
            another_engine=trainer,
        ),
        event_name=Events.EPOCH_COMPLETED,
    )

    trainer.run(train_data_iter, max_epochs=100)
