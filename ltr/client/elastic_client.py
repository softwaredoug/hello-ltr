import os
import requests
from time import perf_counter
from collections import Counter
import concurrent.futures

from .base_client import BaseClient
from ltr.helpers.handle_resp import resp_msg

import elasticsearch.helpers
from elasticsearch.exceptions import NotFoundError
import json
from elasticsearch import Elasticsearch


version_field = 'enrich_version'


class ElasticResp():
    def __init__(self, resp):
        self.status_code = 400
        if 'acknowledged' in resp and resp['acknowledged']:
            self.status_code = 200
        else:
            self.status_code = resp['status']
            self.text = json.dumps(resp, indent=2)

class BulkResp():
    def __init__(self, resp):
        self.status_code = 400
        if resp[0] > 0:
            self.status_code = 201

class SearchResp():
    def __init__(self, resp):
        self.status_code = 400
        if 'hits' in resp:
            self.status_code = 200
        else:
            self.status_code = resp['status']
            self.text = json.dumps(resp, indent=2)


class ElasticClient(BaseClient):
    """ Note on the Elastic client,
        Elastic LTR is not bound to an index like Solr LTR
        so many calls take an index but do not use it

        In the future, we may wish to isolate an Index's feature
        store to a feature store of the same name of the index
    """
    def __init__(self, configs_dir='.'):
        self.docker = os.environ.get('LTR_DOCKER') != None
        self.configs_dir = configs_dir #location of elastic configs

        if self.docker:
            self.host = 'elastic'
        else:
            self.host = 'localhost'

        self.elastic_ep = 'http://{}:9200/_ltr'.format(self.host)
        self.es = Elasticsearch('http://{}:9200'.format(self.host))

    def get_host(self):
        return self.host

    def name(self):
        return "elastic"

    def check_index_exists(self, index):
        return self.es.indices.exists(index=index)

    def delete_index(self, index):
        resp = self.es.indices.delete(index=index, ignore=[400, 404])
        resp_msg(msg="Deleted index {}".format(index), resp=ElasticResp(resp), throw=False)


    def create_index(self, index):
        """ Take the local config files for Elasticsearch for index, reload them into ES"""
        cfg_json_path = os.path.join(self.configs_dir, "%s_settings.json" % index)
        with open(cfg_json_path) as src:
            settings = json.load(src)
            resp = self.es.indices.create(index, body=settings)
            resp_msg(msg="Created index {}".format(index), resp=ElasticResp(resp))


    def to_dataframe(self, index, query={"query": {"match_all": {}}}):
        import pandas as pd
        start = perf_counter()
        docs = []
        for idx, doc in enumerate(elasticsearch.helpers.scan(self.es, index=index, scroll='5m',
                                                              size=1000,
                                                              query=query)):
            docs.append(doc['_source'])
            if idx % 10000 == 0:
                print(f"Scanned {idx} documents -- {perf_counter() - start}")
        return pd.DataFrame(docs)



    def enrich(self, index, enrich_fn, mapping, version, workers=2):
        """Incrementally enrich documents not yet at the specified version."""
        search_scroll_body = {
            "query": {
                "match": {
                    version_field: version - 1
                }
            }
        }
        count = self.es.count(index=index, body=search_scroll_body)
        print(f"Enriching {count['count']} documents in {index}")

        if mapping is not None:
            self.es.indices.put_mapping(index=index, body=mapping)

        def scanner():
            start = perf_counter()
            for idx, doc in enumerate(elasticsearch.helpers.scan(self.es, index=index, scroll='5m',
                                                                  size=1000,
                                                                  query=search_scroll_body)):

                curr_version = int(doc['_source'][version_field])

                assert version == curr_version + 1

                doc['_source'] = enrich_fn(doc['_source'])
                doc["_source"][version_field] = version
                try:
                    yield {
                        "_op_type": "update",
                        "_index": index,
                        "_id": doc['_id'],
                        "doc": doc['_source']
                    }

                    # return self.es.update(index=index, id=doc['_id'], body={"doc": doc['_source']})
                except NotFoundError:
                    print(f"Document {doc['_id']} not found, skipping")

                if idx % 100 == 0:
                    print(f"Enriched {idx} documents -- {perf_counter() - start}")
                    print("--------")

        resps = elasticsearch.helpers.parallel_bulk(self.es, scanner(), thread_count=workers,
                                                    chunk_size=100, request_timeout=120)
        for success, resp in resps:
            if not success:
                print("Failure -- ")
                print(resp)
        self.es.indices.refresh(index=index)




    def index_documents(self, index, doc_src):

        def bulkDocs(doc_src):
            for doc in doc_src:
                if 'id' not in doc:
                    raise ValueError("Expecting docs to have field 'id' that uniquely identifies document")
                doc[version_field] = 0
                addCmd = {"_index": index,
                          "_id": doc['id'],
                          "_source": doc}
                yield addCmd

        resp = elasticsearch.helpers.bulk(self.es, bulkDocs(doc_src), chunk_size=100, request_timeout=120)
        self.es.indices.refresh(index=index)
        resp_msg(msg="Streaming Bulk index DONE {}".format(index), resp=BulkResp(resp))

    def reset_ltr(self, index):
        resp = requests.delete(self.elastic_ep)
        resp_msg(msg="Removed Default LTR feature store".format(), resp=resp, throw=False)
        resp = requests.put(self.elastic_ep)
        resp_msg(msg="Initialize Default LTR feature store".format(), resp=resp)

    def create_featureset(self, index, name, ftr_config):
        resp = requests.post('{}/_featureset/{}'.format(self.elastic_ep, name), json=ftr_config)
        resp_msg(msg="Create {} feature set".format(name), resp=resp)

    def get_feature_name(self, config, ftr_idx):
        return config["featureset"]["features"][int(ftr_idx) - 1]["name"]


    def log_query(self, index, featureset, ids, params={}):
        params = {
            "query": {
                "bool": {
                    "filter": [
                        {
                            "sltr": {
                                "_name": "logged_features",
                                "featureset": featureset,
                                "params": params
                            }
                        }
                    ]
                }
            },
            "ext": {
                "ltr_log": {
                    "log_specs": {
                        "name": "ltr_features",
                        "named_query": "logged_features"
                    }
                }
            },
            "size": 1000
        }

        terms_query = [
            {
                "terms": {
                    "_id": ids
                }
            }
        ]

        if ids is not None:
            params["query"]["bool"]["must"] = terms_query

        resp = self.es.search(index=index, body=params)
        # resp_msg(msg="Searching {} - {}".format(index, str(terms_query)[:20]), resp=SearchResp(resp))

        matches = []
        for hit in resp['hits']['hits']:
            hit['_source']['ltr_features'] = []

            for feature in hit['fields']['_ltrlog'][0]['ltr_features']:
                value = 0.0
                if 'value' in feature:
                    value = feature['value']

                hit['_source']['ltr_features'].append(value)

            matches.append(hit['_source'])

        return matches

    def submit_model(self, featureset, index, model_name, model_payload):
        model_ep = "{}/_model/".format(self.elastic_ep)
        create_ep = "{}/_featureset/{}/_createmodel".format(self.elastic_ep, featureset)

        resp = requests.delete('{}{}'.format(model_ep, model_name))
        print('Delete model {}: {}'.format(model_name, resp.status_code))

        resp = requests.post(create_ep, json=model_payload)
        resp_msg(msg="Created Model {}".format(model_name), resp=resp)

    def submit_ranklib_model(self, featureset, index, model_name, model_payload):
        params = {
            'model': {
                'name': model_name,
                'model': {
                    'type': 'model/ranklib',
                    'definition': model_payload
                }
            }
        }
        self.submit_model(featureset, index, model_name, params)

    def model_query(self, index, model, model_params, query):
        params = {
            "query": query,
            "rescore": {
                "window_size": 1000,
                "query": {
                    "rescore_query": {
                        "sltr": {
                            "params": model_params,
                            "model": model
                        }
                    }
                }
            },
            "size": 1000
        }

        resp = self.es.search(index=index, body=params)
        # resp_msg(msg="Searching {} - {}".format(index, str(query)[:20]), resp=SearchResp(resp))

        # Transform to consistent format between ES/Solr
        matches = []
        for hit in resp['hits']['hits']:
            match = hit['_source']
            match['score'] = hit['_score']
            matches.append(match)

        return matches

    def query(self, index, query):
        resp = self.es.search(index=index, body=query)
        # resp_msg(msg="Searching {} - {}".format(index, str(query)[:20]), resp=SearchResp(resp))

        # Transform to consistent format between ES/Solr
        matches = []
        for hit in resp['hits']['hits']:
            hit['_source']['_score'] = hit['_score']
            matches.append(hit['_source'])

        return matches

    def feature_set(self, index, name):
        resp = requests.get('{}/_featureset/{}'.format(self.elastic_ep,
                                                      name))

        jsonResp = resp.json()
        if not jsonResp['found']:
            raise RuntimeError("Unable to find {}".format(name))

        resp_msg(msg="Fetched FeatureSet {}".format(name), resp=resp)

        rawFeatureSet = jsonResp['_source']['featureset']['features']

        mapping = []
        for feature in rawFeatureSet:
            mapping.append({'name': feature['name']})

        return mapping, rawFeatureSet

    def get_doc(self, doc_id, index):
        resp = self.es.get(index=index, id=doc_id)
        #resp_msg(msg="Fetched Doc".format(docId), resp=ElasticResp(resp), throw=False)
        return resp['_source']

