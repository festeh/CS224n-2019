# NMT with Subword Information assignment

This is an implementation of Assignment 5 from
CS224n (Winter 2019) edition. Most of the code (data preprocessing, training loop) was written from scratch and uses
NLP *Best Practices*™:

* **SpaCy** for text preprocessing. Spacy has tokenizers for English and Spanish languages
* **Allennlp** for text vectorization. It's helpful in this assignment as it helps to easily add character
    indexer along with word indexer
* **Einops** for tensor manipulations. Suppose you have tensor `input` with dimensions

    `batch_size X max_num_words_sentence X max_word_length X char_emb_dim` 
    
    and you want
    to construct word embeddings from it. The first step would be to pass
    it into Conv1d, that requires 3d tensor. So instead of rather cumbersome line
    
    `input.view(input.size(0) * input.size(1), input.size(2), -1).permute(0, 2, 1)` 
    
    with Einops you can simply write more readable expression 

    `einops.rearrange(input, bs mw ml e -> (bs mw) e ml)` 
    
* **Pytorch-Ignite** for creating easily extensible training loop. With it you can start with simple loop that only
       optimizes model weights and then enhance it with callbacks that for example compute additional metrics, 
       log results to Tensorboard and save model weigths

# How to run

Place assigment data in the `en_es_data` directory and then run
```bash
export PYTHONPATH="../"
conda env create -f environment.yml
conda activate CS224n
python run.py
python model_test.py
```
   
# TODO

* Several small experiments need to be carried out
* Fallback char-level word model for UNK tokens is not implemented (doubt it will increase BLEU score much as UNK tokens
    are rarely produced) 
* Written part is in progress
 
# Results
The current BLEU* score is **24.6**

*As SpaCy tokenizers are used here, BLEU score is computed
with slightly different reference data (previously it was tokenized naively), so it cannot be directly compared with result
from Assignment 4.
