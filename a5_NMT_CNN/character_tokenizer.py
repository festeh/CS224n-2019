"""Borrowed from allennlp with small tweak: max length is restricted"""
from logging import info
from typing import List

from allennlp.data import Tokenizer, Token
from overrides import overrides


class MyCharacterTokenizer(Tokenizer):
    def __init__(self,
                 lowercase_characters: bool = False,
                 max_length=None,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        self.max_length = max_length
        self._lowercase_characters = lowercase_characters
        self._start_tokens = start_tokens or []
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        if self._lowercase_characters:
            text = text.lower()
        tokens = [Token(t) for t in list(text)]
        if self.max_length is not None:
            if len(tokens) > self.max_length:
                info(f"Too long word: {text}")
            tokens = tokens[:self.max_length]
        for start_token in self._start_tokens:
            if isinstance(start_token, int):
                token = Token(text_id=start_token, idx=0)
            else:
                token = Token(text=start_token, idx=0)
            tokens.insert(0, token)
        for end_token in self._end_tokens:
            if isinstance(end_token, int):
                token = Token(text_id=end_token, idx=0)
            else:
                token = Token(text=end_token, idx=0)
            tokens.append(token)
        return tokens