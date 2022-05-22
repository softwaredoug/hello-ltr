"""Add USE for entire raw text to vmware corpus."""
import tensorflow_text
import tensorflow_hub as hub

_use_mapping = {
  "properties": {
    "raw_text_use": {
        "type": "dense_vector",
        "dims": 512
    }
  }
}

_use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")


def _process_use(doc_source):
    """Process USE data"""
    doc_source['raw_text_use'] = _use(doc_source['raw_text']).numpy().tolist()[0]
    return doc_source

mapping = _use_mapping
enrichment = _process_use
