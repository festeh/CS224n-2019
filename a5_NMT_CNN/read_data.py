from allennlp.data import DatasetReader, Instance
from typing import Iterable, List, Iterable, List

from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import TokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer, CharacterTokenizer
from allennlp.data.tokenizers.word_splitter import WordSplitter, SpacyWordSplitter


@DatasetReader.register("nmt-dataset")
class NMTDataReader(DatasetReader):
    def __init__(self, convert_to_lowercase=True):
        super().__init__(lazy=False)
        self.convert_to_lowercase = convert_to_lowercase
        # TODO: change to es_core_news_sm when you'll finish debugging!!
        self.source_tokenizer = WordTokenizer(SpacyWordSplitter("en_core_web_sm"))
        self.target_tokenizer = WordTokenizer(
            SpacyWordSplitter("en_core_web_sm"), start_tokens=["BOS"], end_tokens=["EOS"]
        )

        # TODO: truncate words to 21 characters
        #
        # TODO: pass lowercase argument
        self.source_token_indexers = {
            "token_characters": TokenCharactersIndexer(
                "char_src", min_padding_length=5
            ),
            "tokens": SingleIdTokenIndexer("token_src"),
        }
        self.target_token_indexers = {
            "token_characters": TokenCharactersIndexer(
                "char_trg",
                min_padding_length=5,
            ),
            "token_characters_output": TokenCharactersIndexer(
                "char_trg",
                min_padding_length=5,
                character_tokenizer=CharacterTokenizer(start_tokens=["SOT"], end_tokens=["EOT"]),  # lul
            ),
            "tokens": SingleIdTokenIndexer("token_trg"),
        }

    def text_to_instance(self, source_sentence, target_sentence) -> Instance:
        fields = {
            "source_sentence": TextField(source_sentence, self.source_token_indexers),
            "target_sentence": TextField(target_sentence, self.target_token_indexers),
        }
        return Instance(fields)

    def _read(self, file_paths: List) -> Iterable[Instance]:
        source_file_path, target_file_path = file_paths
        with open(source_file_path) as f, open(target_file_path) as ff:
            for source_line, target_line in zip(f.readlines(), ff.readlines()):
                source_tokenized = self.source_tokenizer.tokenize(source_line)
                target_tokenized = self.target_tokenizer.tokenize(target_line)
                yield self.text_to_instance(source_tokenized, target_tokenized)


if __name__ == "__main__":
    reader = NMTDataReader()
    source_path = "a5_NMT_CNN/en_es_data/sample.es"
    target_path = "a5_NMT_CNN/en_es_data/sample.en"
    instances = reader.read([source_path, target_path])
    print(instances[0]["source_sentence"])
