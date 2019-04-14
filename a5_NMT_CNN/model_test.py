import pickle

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import TokenCharactersIndexer
from ignite.utils import apply_to_tensor, apply_to_type
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from a5_NMT_CNN.character_tokenizer import MyCharacterTokenizer
from a5_NMT_CNN.nmt_model import NMTModel
from a5_NMT_CNN.read_data import NMTDataReader

config = Params.from_file("config.jsonnet")
# config["device"] = "cpu"
with open(config.pop("test_instances_path"), "rb") as f:
    instances = pickle.load(f)
vocab = Vocabulary.from_files(config.pop("vocab_path"))
model = NMTModel(config, vocab)
model.eval()
model.load_state_dict(torch.load("my_model/nmt_mymodel_28.pth"))
model.to(model.device)
test_iterator = BasicIterator(batch_size=1)
test_iterator.index_with(vocab)

all_hyps = []
all_refs = []

for idx, test_data_sample in tqdm(
    enumerate(test_iterator(instances, shuffle=False, num_epochs=1)),
    total=len(instances),
):
    # noinspection PyTypeChecker
    with torch.no_grad():
        hyps = model.beam_search(
            apply_to_type(test_data_sample, torch.Tensor, lambda t: t.to(model.device)),
            TokenCharactersIndexer(
                "char_trg",
                character_tokenizer=MyCharacterTokenizer(max_length=21),
            )
        )
    best_hyp = hyps[0].value
    # try:
    ref = [str(t) for t in instances[idx]["target_sentence"].tokens[1:-1]]
    all_hyps.append(best_hyp)
    all_refs.append([ref])
    if idx % 100 == 0:
        print(" ".join(best_hyp), "->", " ".join(ref))
    # except:
    #     pass
    # if idx == 100:
    #     break
print(corpus_bleu(all_refs, all_hyps) * 100)
