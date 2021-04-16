# Standard libraries
from typing import List

# Third-party libraries
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


def _plot_2d(embedded_vocab: np.ndarray, vocab: List[str]) -> None:
    pca = PCA(n_components=2)
    results = pca.fit_transform(embedded_vocab)
    plt.figure(figsize=(15, 12))
    plt.scatter(results[:, 0], results[:, 1])
    for i, word in enumerate(vocab):
        plt.annotate(word, xy=(results[i, 0], results[i, 1]))
    plt.show()


def _plot_3d(embedded_vocab: np.ndarray, vocab: List[str]) -> None:
    pca = PCA(n_components=2)
    results = pca.fit_transform(embedded_vocab)
    plt.figure(figsize=(15, 12))
    ax = plt.axes(projection="3d")
    ax.scatter(results[:, 0], results[:, 1], results[:, 2], c="b")
    for i, word in enumerate(vocab):
        ax.text(results[i, 0], results[i, 1], results[i, 2], word, c="k")
    plt.show()


def _visualize_embedding(
    embedded_vocab: np.ndarray,
    vocab: List[str],
    number_of_projections: int = 30,
    dimensions_to_project: int = 3,
) -> None:
    vocab_index_to_keep = np.random.randint(0, len(vocab), size=number_of_projections)
    extracted_vocab = vocab[vocab_index_to_keep]
    if dimensions_to_project == 3:
        _plot_3d(embedded_vocab, extracted_vocab)
    elif dimensions_to_project == 2:
        _plot_2d(embedded_vocab, extracted_vocab)
    else:
        raise TypeError(f"{dimensions_to_project} is not supported, use 2 or 3 instead")


def visualize_embedding_from_gensim_word2vec_model(
    model, number_of_projections: int = 30, dimensions_to_project: int = 3
) -> None:

    if not isinstance(model, Word2Vec):
        raise TypeError(
            f"{type(model)} is not supported by this function, model must be of type gensim.models.Word2Vec"
        )

    vocab: List[str] = model.wv.vocab
    embedded_vocab: np.ndarray = model[
        vocab
    ]  # shape of embedded_vocab must be (len(vocab), embedding space dimension)
    _visualize_embedding(
        embedded_vocab,
        vocab,
        number_of_projections=number_of_projections,
        dimensions_to_project=dimensions_to_project,
    )
