from collections import KeysView

from allennlp.data import Vocabulary
from ignite.metrics import Metric
from nltk.translate.bleu_score import corpus_bleu
from numpy import exp


class Loss(Metric):
    def reset(self):
        self.cum_loss = 0
        self.n_examples = 0

    def update(self, output):
        _, logprobs = output
        self.cum_loss -= logprobs.sum()
        self.n_examples += logprobs.size(0)

    def compute(self):
        return self.cum_loss / self.n_examples


class Perplexity(Metric):
    def reset(self):
        self.cum_loss = 0
        self.cum_tgt_words = 0

    def update(self, output):
        _, logprobs = output
        self.cum_loss -= logprobs.sum()
        # create binary mask, skip one token
        self.cum_tgt_words += ((logprobs != 0).sum(dim=1) - 1).sum()

    def compute(self):
        return exp(self.cum_loss / self.cum_tgt_words)


def truncate_hyp(hyp):
    try:
        idx = hyp.index("EOS")
        return hyp[:idx]
    except ValueError:
        return hyp


class BLEU(Metric):
    def __init__(self, vocab: Vocabulary, instances):
        super().__init__()
        refs = [
            [t.text for t in inst["target_sentence"].tokens[1:-1]] for inst in instances
        ]

        self.orig_refs = refs
        self.refs = refs
        self.vocab = vocab

    def reset(self):
        self.refs = self.orig_refs
        self.bleu = 0

    def update(self, output):
        logits, _ = output
        hyps = output
        batch_size = logits.size(0)
        hyps_idxs = logits.argmax(-1).cpu().numpy()
        hyps = [[self.vocab.get_token_from_index(idx, "token_trg") for idx in idxs]
                for idxs in hyps_idxs]
        hyps = [truncate_hyp(hyp) for hyp in hyps]
        refs = self.refs[:batch_size]
        print("\n".join([" ".join(hyp) for hyp in hyps][:3]),
              "\n".join([" ".join(ref) for ref in refs][:3]))
        self.refs = self.refs[batch_size:]
        self.bleu += corpus_bleu([[ref] for ref in refs], hyps)

    def compute(self):
        return self.bleu * 100
