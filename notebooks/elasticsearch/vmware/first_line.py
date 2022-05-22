_first_line_mapping = {
  "properties": {
    "first_line": {
    "type": "text",
    "analyzer": "content_analyzer",
    "fields": {
      "bigrams": {
        "type": "text",
        "analyzer": "content_bigrams"
      }
    }
  }
  }
}


def _add_first_line(doc_source):
    """First lines appear to be titles?"""
    doc_source['first_line'] = doc_source['raw_text'].split('\n')[0]
    return doc_source


mapping = _first_line_mapping
enrichment = _add_first_line
