early_long_remaining_lines_mapping = {
  "properties": {
        "first_remaining_lines": {
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

def process_first_n_remaining_lines(doc_source):
    """Process USE data on long passages and the first line."""
    long_remaining_lines = [line for line in doc_source['remaining_lines'] if len(line) > 20][:10]
    doc_source["first_remaining_lines"] = long_remaining_lines
    return doc_source

mapping = early_long_remaining_lines_mapping
enrichment = process_first_n_remaining_lines
