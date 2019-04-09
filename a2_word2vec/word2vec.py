#!/usr/bin/env python
import random

import numpy as np

from a2_word2vec.utils.gradcheck import gradcheck_naive
from a2_word2vec.utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """
    return np.exp(-np.logaddexp(0, -x))


def naive_softmax_loss_and_gradient(
    center_word_vec, outside_word_idx, outside_vectors, dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    center_word_vec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outside_word_idx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outside_vectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    grad_center_vec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    grad_outside_vecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """
    probs_context_vectors = softmax(outside_vectors.dot(center_word_vec))
    loss = -np.log(probs_context_vectors[outside_word_idx])

    probs_context_vectors[outside_word_idx] -= 1

    grad_center_vec = outside_vectors.T.dot(probs_context_vectors)
    grad_outside_vecs = (
        np.tile(center_word_vec, (len(outside_vectors), 1))
        * probs_context_vectors[:, None]
    )
    return loss, grad_center_vec, grad_outside_vecs


def get_negative_samples(outside_word_idx, dataset, K):
    """ Samples K indexes which are not the outside_word_idx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outside_word_idx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def neg_sampling_loss_and_gradient(
    center_word_vec, outside_word_idx, outside_vectors, dataset, K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naive_softmax_loss_and_gradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    neg_sample_word_indices = get_negative_samples(outside_word_idx, dataset, K)
    indices = [outside_word_idx] + neg_sample_word_indices

    window_outside_vectors = outside_vectors[indices]

    scores = window_outside_vectors.dot(center_word_vec)
    scores[1:] *= -1
    probs = sigmoid(scores)

    loss = -np.sum(np.log(probs))
    weights = 1 - probs
    weights[0] *= -1
    grad_center_vec = window_outside_vectors.T.dot(weights)

    grad_outside_vecs = np.zeros_like(outside_vectors)

    np.add.at(
        grad_outside_vecs,
        indices,
        np.tile(center_word_vec, (len(window_outside_vectors), 1)) * weights[:, None],
    )

    return loss, grad_center_vec, grad_outside_vecs


def skipgram(
    current_center_word,
    windowSize,
    outside_words,
    word2Ind,
    center_word_vectors,
    outside_vectors,
    dataset,
    word2vecLossAndGradient=naive_softmax_loss_and_gradient,
):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    current_center_word -- a string of the current center word
    windowSize -- integer, context window size
    outside_words -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    center_word_vectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
    outside_vectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vectors
            (dJ / dV in the pdf handout)
    gradOutsideVectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
    """
    center_word_index = word2Ind[current_center_word]
    center_word_vector = center_word_vectors[center_word_index]

    loss = 0.0
    grad_center_vecs = np.zeros(center_word_vectors.shape)
    grad_outside_vectors = np.zeros(outside_vectors.shape)

    for outside_word in outside_words:
        cur_loss, cur_grad_center_vec, cur_grad_outside_vecs = word2vecLossAndGradient(
            center_word_vec=center_word_vector,
            outside_word_idx=word2Ind[outside_word],
            outside_vectors=outside_vectors,
            dataset=dataset,
        )
        loss += cur_loss
        grad_center_vecs[center_word_index] += cur_grad_center_vec
        grad_outside_vectors += cur_grad_outside_vecs

    return loss, grad_center_vecs, grad_outside_vectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################


def word2vec_sgd_wrapper(
    word2vec_model,
    word2_ind,
    word_vectors,
    dataset,
    window_size,
    word2vec_loss_and_gradient=naive_softmax_loss_and_gradient,
):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(word_vectors.shape)
    N = word_vectors.shape[0]
    center_word_vectors = word_vectors[: int(N / 2), :]
    outside_vectors = word_vectors[int(N / 2) :, :]
    for i in range(batchsize):
        windowSize1 = random.randint(1, window_size)
        center_word, context = dataset.get_random_context(windowSize1)

        c, gin, gout = word2vec_model(
            center_word,
            windowSize1,
            context,
            word2_ind,
            center_word_vectors,
            outside_vectors,
            dataset,
            word2vec_loss_and_gradient,
        )
        loss += c / batchsize
        grad[: int(N / 2), :] += gin / batchsize
        grad[int(N / 2) :, :] += gout / batchsize

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type("dummy", (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return (
            tokens[random.randint(0, 4)],
            [tokens[random.randint(0, 4)] for i in range(2 * C)],
        )

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.get_random_context = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    print("==== Gradient check for skip-gram with naive_softmax_loss_and_gradient ====")
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(
            skipgram, dummy_tokens, vec, dataset, 5, naive_softmax_loss_and_gradient
        ),
        dummy_vectors,
        "naive_softmax_loss_and_gradient Gradient",
    )

    print("==== Gradient check for skip-gram with neg_sampling_loss_and_gradient ====")
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(
            skipgram, dummy_tokens, vec, dataset, 5, neg_sampling_loss_and_gradient
        ),
        dummy_vectors,
        "neg_sampling_loss_and_gradient Gradient",
    )
    #
    print("\n=== Results ===")
    print("Skip-Gram with naive_softmax_loss_and_gradient")
    #
    print("Your Result:")
    print(
        "Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            *skipgram(
                "c",
                3,
                ["a", "b", "e", "d", "b", "c"],
                dummy_tokens,
                dummy_vectors[:5, :],
                dummy_vectors[5:, :],
                dataset,
            )
        )
    )
    #
    print("Skip-Gram with neg_sampling_loss_and_gradient")
    print("Your Result:")
    print(
        "Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            *skipgram(
                "c",
                1,
                ["a", "b"],
                dummy_tokens,
                dummy_vectors[:5, :],
                dummy_vectors[5:, :],
                dataset,
                neg_sampling_loss_and_gradient,
            )
        )
    )
    print("Expected Result: Value should approximate these:")
    print(
        """Loss: 16.15119285363322
    Gradient wrt Center Vectors (dJ/dV):
     [[ 0.          0.          0.        ]
     [ 0.          0.          0.        ]
     [-4.54650789 -1.85942252  0.76397441]
     [ 0.          0.          0.        ]
     [ 0.          0.          0.        ]]
     Gradient wrt Outside Vectors (dJ/dU):
     [[-0.69148188  0.31730185  2.41364029]
     [-0.22716495  0.10423969  0.79292674]
     [-0.45528438  0.20891737  1.58918512]
     [-0.31602611  0.14501561  1.10309954]
     [-0.80620296  0.36994417  2.81407799]]
        """
    )


#
if __name__ == "__main__":
    test_word2vec()
