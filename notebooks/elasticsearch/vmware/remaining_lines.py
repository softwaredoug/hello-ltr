_remaining_lines_mapping = {
  "properties": {
    "remaining_lines": {
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


def _add_remaining_lines(doc_source):
    """First lines appear to be titles?"""
    doc_source['remaining_lines'] = doc_source['raw_text'].split('\n')[1:]
    return {'remaining_lines': doc_source['remaining_lines']}

mapping = _remaining_lines_mapping
enrichment = _add_remaining_lines
