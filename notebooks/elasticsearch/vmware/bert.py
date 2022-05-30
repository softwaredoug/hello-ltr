# Code based on https://medium.com/swlh/fine-tuning-bert-for-text-classification-and-question-answering-using-tensorflow-framework-4d09daeb3330
#
# https://github.com/dredwardhyde/bert-examples/blob/main/bert_squad_tensorflow.py
#
# Notes on BERT
#
#  This model trains a (pretrained?) BERT model on the SQuAD dataset.
#
#  It expects that the model
#


# Reuse a BERT model
# Based on https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/bert_experts.ipynb#scrollTo=GXrSO2Vc1Qtr


# Questions
# What are the BERT inputs

# 3 numpy arrays
#   * input size * 128 (number of tokens?)
#
#
# - Masks?                input size * 128
#      Whether input is available in the token?
#
# - Input Type Ids:
#      Seems to always be 0?
#
# - Input token IDs?      input
#       The token IDs for the input.
#       101 - start
#       102 - end
#         0 - none
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import tensorflow_text as text  # Imports TF ops for preprocessing.

BERT_MODEL = "https://tfhub.dev/google/experts/bert/wiki_books/2"
PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

_preprocess = hub.load(PREPROCESS_MODEL)
_bert = hub.load(BERT_MODEL)


_bert_mapping = {
  "properties": {
    "first_line_bert": {
        "type": "dense_vector",
        "dims": 768
    },
    "long_remaining_lines_bert_0": {
        "type": "dense_vector",
        "dims": 768
    },
    "long_remaining_lines_bert_1": {
        "type": "dense_vector",
        "dims": 768
    },
    "long_remaining_lines_bert_2": {
        "type": "dense_vector",
        "dims": 768
    },
    "long_remaining_lines_bert_3": {
        "type": "dense_vector",
        "dims": 768
    },
    "long_remaining_lines_bert_4": {
        "type": "dense_vector",
        "dims": 768
    },
    "long_remaining_lines_bert_5": {
        "type": "dense_vector",
        "dims": 768
    },
    "long_remaining_lines_bert_6": {
        "type": "dense_vector",
        "dims": 768
    },
    "long_remaining_lines_bert_7": {
        "type": "dense_vector",
        "dims": 768
    },
    "long_remaining_lines_bert_8": {
        "type": "dense_vector",
        "dims": 768
    },
    "long_remaining_lines_bert_9": {
        "type": "dense_vector",
        "dims": 768
    }
  }
}


# should make this a context manager
class SentenceCache:

    def __init__(self):
        self.f = open("sentences.txt", "w")

    def add_sentence(self, sentences):
        self.f.write('\n'.join(sentences))

    def end_document(self):
        self.f.write('---\n')

    def write_sentences(self):
        self.f.write('\n')
        self.f.close()
        #print(f"Writing {len(self.all_sentences)} sentences to file")
        #with open("sentences.txt", "w") as f:
        #    for sentence in self.all_sentences:
        #        f.write(sentence)

    def done(self):
        self.f.close()

    def read_sentences(self):
        return [sentence for sentence in open("sentences.txt", "r").readlines()
                if sentence != '---\n']

def sentences_embedding(sentences):
    """Get the embeddings of the start (CLS) token."""
    inputs = _preprocess(sentences)
    all_sentences.append(sentences)
    embeddings = _bert(inputs)['pooled_output']
    return embeddings.numpy().tolist()

def _process_bert_remaining_lines(doc_source):
    """Process USE data on long passages and the first line."""
    doc_source["first_line_bert"] = sentences_embedding([doc_source["first_line"]])[0]
    assert len(doc_source["first_line_bert"]) == 768
    long_remaining_lines = [line for line in doc_source['remaining_lines'] if len(line) > 20][:10]
    remaining_lines = sentences_embedding(long_remaining_lines)
    for idx, line in enumerate(remaining_lines):
        if idx < 10:
            doc_source[f"long_remaining_lines_bert_{idx}"] = line
            assert len(doc_source[f"long_remaining_lines_bert_{idx}"]) == 768
    return doc_source


cache = SentenceCache()


def add_sentences(doc_source):
    cache.add_sentence([doc_source["first_line"]])
    long_remaining_lines = [line for line in doc_source['remaining_lines'] if len(line) > 20]
    cache.add_sentence(long_remaining_lines)
    cache.end_document()
    return doc_source


def bert_all():
    sentences = cache.read_sentences()
    import pdb; pdb.set_trace()
    embeddings = sentences_embedding(sentences)
    all_embeddings = []
    for sentence, embed in zip(senteces, embeddings):
        all_embeddings.append({'sentence': sentence, 'embedding': embed})
    all_embeddings = pd.DataFrame(all_embeddings)
    all_embeddings.to_pickle(all_embeddings, "all_embeddings.pkl")



preprocess = add_sentences

enrichment = _process_bert_remaining_lines
mapping = _bert_mapping
