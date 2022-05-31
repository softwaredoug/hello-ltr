_three_lines_mapping = {
  "properties": {
    "three_lines_appended": {
        "type": "text",
        "analyzer": "content_analyzer",
      },
    "sixty_token_strides": {
        "type": "text",
        "analyzer": "content_analyzer",
      }
  }
}


def _three_lines_together(doc_source, stride_start=20, stride_size=60):
    """First lines appear to be titles?"""
    lines = doc_source['raw_text'].split('\n')
    doc_source['three_lines_appended'] = []
    if len(lines) > 2:
        for line1, line2, line3 in zip(lines, lines[1:], lines[2:]):
            doc_source['three_lines_appended'].append(line1 + "\n" + line2 + "\n" + line3)
    else:
        doc_source['three_lines_appended'] = [doc_source['raw_text']]

    doc_source['sixty_token_strides'] = []
    tokens = " ".join(lines).split()
    for start in range(0, len(tokens), stride_start):
        stride = tokens[start:start+stride_size]
        assert len(stride) <= stride_size
        doc_source['sixty_token_strides'].append(" ".join(stride))
    return doc_source


mapping = _three_lines_mapping
enrichment = _three_lines_together
