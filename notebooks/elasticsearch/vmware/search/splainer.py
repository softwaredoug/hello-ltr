import urllib.parse
import json


def splainer_url(splainer='http://splainer.io/#/es_',
                 es_url='http://localhost:9200',
                 es_body={},
                 es_index='vmware',
                 title_field='first_line',
                 body_fields=['raw_text']):
    field_spec = urllib.parse.quote_plus(
        f"title:{title_field} {' '.join(body_fields)}")
    es_body = urllib.parse.quote_plus(json.dumps(es_body))
    es_url = urllib.parse.quote(f"{es_url}/{es_index}/_search")

    url_params = f"?esUrl={es_url}&esQuery={es_body}&fieldSpec={field_spec}"

    return splainer + url_params


def splainer_open(splainer_url):
    import webbrowser
    webbrowser.open(splainer_url)
