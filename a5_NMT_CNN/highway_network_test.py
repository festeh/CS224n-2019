from allennlp.common import Params
import logging

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss
from torch.nn import Module, Linear
from torch.nn.functional import nll_loss, cross_entropy
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm

from a5_NMT_CNN.highway import HighwayNetwork


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST(download=True, root=".", transform=data_transform, train=True),
        batch_size=train_batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        MNIST(download=False, root=".", transform=data_transform, train=False),
        batch_size=val_batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


class DummyNetwork(Module):
    def __init__(self, hw_network):
        super().__init__()
        self.hw_network = hw_network
        self.out_layer = Linear(784, 10)

    def forward(self, input_):
        return self.out_layer(self.hw_network(input_.view(input_.size(0), -1)))


def run():

    train_batch_size = 64
    val_batch_size = 64
    lr = 0.001
    momentum = 0.01
    log_interval = 10
    epochs = 5

    device = "cuda"
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = HighwayNetwork(Params({"dropout_rate": 0, "input_size": 784})).to(device)
    predictor = DummyNetwork(hw_network=model).to(device)
    optimizer = SGD(predictor.parameters(), lr=lr, momentum=momentum)
    trainer = create_supervised_trainer(predictor, optimizer, cross_entropy, device=device)
    evaluator = create_supervised_evaluator(
        predictor, metrics={"accuracy": Accuracy(), "nll": Loss(cross_entropy)}, device=device
    )

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=desc.format(0))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                engine.state.epoch, avg_accuracy, avg_nll
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                engine.state.epoch, avg_accuracy, avg_nll
            )
        )

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
    # model = HighwayNetwork(Params({"dropout_rate": 0, "input_size": 764})).cuda()
    # predictor = DummyNetwork(hw_network=model).cuda()
