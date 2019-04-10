#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""
import sys
from collections import namedtuple
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.nn.util import sort_batch_by_length
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from a4_NMT_RNN.model_embeddings import ModelEmbeddings
from a5_NMT_CNN.embedder import NMTEmbedder

Hypothesis = namedtuple("Hypothesis", ["value", "score"])


class NMTModel(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """

    def __init__(self, config: Params, vocab: Vocabulary):
        super().__init__()
        # embed_size, hidden_size, vocab, dropout_rate = 0.2
        self.source_embedder = NMTEmbedder(
            vocab.get_vocab_size("char_src"), config.duplicate()
        )
        self.target_embedder = NMTEmbedder(
            vocab.get_vocab_size("char_trg"), config.duplicate()
        )

        self.hidden_size = config.pop("hidden_size")
        self.dropout_rate = config.pop("dropout_rate")
        self.vocab = vocab
        self.target_vocab_size = self.vocab.get_vocab_size("token_trg")
        self.device = config.pop("device")

        self.encoder = nn.LSTM(
            input_size=self.source_embedder.char_emb_size,
            hidden_size=self.hidden_size,
            bidirectional=True,
            bias=True,
            batch_first=True,
        )
        self.decoder = nn.LSTMCell(
            input_size=self.source_embedder.char_emb_size + self.hidden_size,
            hidden_size=self.hidden_size,
            bias=True,
        )
        self.h_projection = nn.Linear(
            in_features=self.hidden_size * 2, out_features=self.hidden_size, bias=False
        )
        self.c_projection = nn.Linear(
            in_features=self.hidden_size * 2, out_features=self.hidden_size, bias=False
        )
        self.att_projection = nn.Linear(
            in_features=self.hidden_size * 2, out_features=self.hidden_size, bias=False
        )
        self.combined_output_projection = nn.Linear(
            in_features=self.hidden_size * 3, out_features=self.hidden_size, bias=False
        )
        self.target_vocab_projection = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.target_vocab_size,
            bias=False,
        )
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, data) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # encode input data
        print()
        enc_hiddens, dec_init_state, enc_masks = self.encode(
            data["source_sentence"]["token_characters"]
        )
        combined_outputs = self.decode(
            enc_hiddens, enc_masks, dec_init_state, data["target_sentence"]
        )

        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        target_word_idxs = data["target_sentence"]["tokens"][:, 1:]
        target_masks = (target_word_idxs != 0).float()

        target_gold_words_log_prob = torch.gather(
            P, index=target_word_idxs.unsqueeze(-1), dim=-1
        ).squeeze(-1) * target_masks

        return target_gold_words_log_prob.sum(dim=1)

    def encode(
        self, source_batch
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        input_embedded = self.source_embedder(source_batch)

        enc_masks = source_batch.sum(dim=2) != 0
        lengths = enc_masks.sum(dim=1)

        input_embedded_sorted, new_lengths, rest_idxs, _ = sort_batch_by_length(
            input_embedded, lengths
        )

        # TODO: remove assertion
        assert torch.equal(input_embedded, input_embedded_sorted[rest_idxs])

        enc_hiddens, (last_hidden, last_cell) = self.encoder(
            pack_padded_sequence(input_embedded_sorted, new_lengths, batch_first=True)
        )
        enc_hiddens, _ = pad_packed_sequence(
            enc_hiddens, padding_value=0, batch_first=True
        )

        # restore original ordering
        enc_hiddens = enc_hiddens[rest_idxs]
        last_hidden = last_hidden[:, rest_idxs, :]
        last_cell = last_cell[:, rest_idxs, :]

        init_decoder_hidden = self.h_projection(
            torch.cat([last_hidden[0], last_hidden[1]], dim=1)
        )
        init_decoder_cell = self.c_projection(
            torch.cat([last_cell[0], last_cell[1]], dim=1)
        )
        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        return enc_hiddens, dec_init_state, enc_masks

    def decode(
        self,
        enc_hiddens: torch.Tensor,
        enc_masks: torch.Tensor,
        dec_init_state: Tuple[torch.Tensor, torch.Tensor],
        target_data: Dict,
    ) -> torch.Tensor:
        """Compute combined output vectors for a batch.
        """
        # Chop of the <END> token for max length sentences.
        target_char_input = target_data["token_characters"]
        target_char_input_embedded = self.target_embedder(target_char_input)[:, :-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        enc_hiddens_proj = self.att_projection(enc_hiddens)
        for Y_t in torch.split(target_char_input_embedded, 1, dim=1):
            Y_t_squeezed = Y_t.squeeze(1)
            Ybar_t = torch.cat([Y_t_squeezed, o_prev], dim=1)
            dec_state, o_t, _ = self.step(
                Ybar_t,
                dec_state=dec_state,
                enc_hiddens=enc_hiddens,
                enc_hiddens_proj=enc_hiddens_proj,
                enc_masks=enc_masks,
            )
            combined_outputs.append(o_t)
            o_prev = o_t
        return torch.stack(combined_outputs, dim=1)

    def step(
        self,
        Ybar_t: torch.Tensor,
        dec_state: Tuple[torch.Tensor, torch.Tensor],
        enc_hiddens: torch.Tensor,
        enc_hiddens_proj: torch.Tensor,
        enc_masks: torch.Tensor,
    ) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length. 

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """
        new_dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = new_dec_state
        e_t = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(2)).squeeze(2)
        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(1 - enc_masks.byte(), -float("inf"))

        alpha_t = nn.functional.softmax(input=e_t, dim=1)
        a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens).squeeze(1)
        U_t = torch.cat([a_t, dec_hidden], dim=1)
        V_t = self.combined_output_projection(U_t)
        O_t = self.dropout(torch.tanh(V_t))
        combined_output = O_t
        return new_dec_state, combined_output, e_t

    def generate_sent_masks(
        self, enc_hiddens: torch.Tensor, source_lengths: List[int]
    ) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size. 
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
        
        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(
            enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float
        )
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def beam_search(
        self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70
    ) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt["</s>"]

        hypotheses = [["<s>"]]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(
                hyp_num, src_encodings.size(1), src_encodings.size(2)
            )

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(
                hyp_num,
                src_encodings_att_linear.size(1),
                src_encodings_att_linear.size(2),
            )

            y_tm1 = torch.tensor(
                [self.vocab.tgt[hyp[-1]] for hyp in hypotheses],
                dtype=torch.long,
                device=self.device,
            )
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _ = self.step(
                x,
                h_tm1,
                exp_src_encodings,
                exp_src_encodings_att_linear,
                enc_masks=None,
            )

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (
                hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t
            ).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
                contiuating_hyp_scores, k=live_hyp_num
            )

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(
                prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores
            ):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == "</s>":
                    completed_hypotheses.append(
                        Hypothesis(value=new_hyp_sent[1:-1], score=cand_new_hyp_score)
                    )
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(
                live_hyp_ids, dtype=torch.long, device=self.device
            )
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(
                new_hyp_scores, dtype=torch.float, device=self.device
            )

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(
                Hypothesis(value=hypotheses[0][1:], score=hyp_scores[0].item())
            )

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    # @property
    # def device(self) -> torch.device:
    #     """ Determine which device to place the Tensors upon, CPU or GPU.
    #     """
    #     return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params["args"]
        model = NMT(vocab=params["vocab"], **args)
        model.load_state_dict(params["state_dict"])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print("save model parameters to [%s]" % path, file=sys.stderr)

        params = {
            "args": dict(
                embed_size=self.model_embeddings.embed_size,
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate,
            ),
            "vocab": self.vocab,
            "state_dict": self.state_dict(),
        }

        torch.save(params, path)
