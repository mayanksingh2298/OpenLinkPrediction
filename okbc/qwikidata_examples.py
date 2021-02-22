
# Get entity ids from api
from qwikidata.linked_data_interface import get_entity_dict_from_api
from qwikidata.entity import WikidataItem, WikidataProperty, WikidataLexeme
from tqdm import tqdm
import pickle
wikidata = pickle.load(open('/home/keshav/olpbench/wikidata_ids.pkl','rb'))
english_labels_nf, items_nf = set(), set()
labelsD = dict()
for item in tqdm(wikidata):
    try:
        entity_dict = get_entity_dict_from_api(item)
    except:
        items_nf.add(item)
        continue
    if 'en' not in entity_dict['labels']:
        english_labels_nf.add(item)
        continue
    labelsD[item] = entity_dict['labels']['en']['value']

# Get entity ids from json dump
import pickle
from tqdm import tqdm
from qwikidata.json_dump import WikidataJsonDump

wjd = WikidataJsonDump("/home/keshav/wikidata-20201109-all.json.bz2")
namesD = dict()
not_found_english_label = 0
for item in tqdm(wjd):
    entity_id =item['id']
    if 'en' not in item['labels']:
        continue
    namesD[entity_id] = item['labels']['en']['value']

import spacy
from spacyEntityLinker import EntityLinker
#Initialize Entity Linker
entityLinker = EntityLinker()
#initialize language model
nlp = spacy.load("en_core_web_sm")
#add pipeline
nlp.add_pipe(entityLinker, last=True, name="entityLinker")
doc = nlp("I watched the Pirates of the Carribean last silvester")
all_linked_entities=doc._.linkedEntities
for sent in doc.sents:
        sent._.linkedEntities.pretty_print()

# Get entity id from sparql
chunk_size = 500                                                                                                                                                        [78/4699]   ...: from tqdm import tqdm
import pickle
from qwikidata.sparql import return_sparql_query_results
wikidata = pickle.load(open('/home/keshav/olpbench/wikidata_ids.pkl','rb'))
wikidata = list(wikidata)
valuesD = dict()
for i in tqdm(range(0, len(wikidata), chunk_size)):
    values = wikidata[i:i+chunk_size]
    final_values = []
    values_str = ''
    for value in values:
        if value.startswith('Q'):
            values_str += 'wd:'+value+' '
            final_values.append(value)
    query_string = """
    SELECT ?item ?itemLabel WHERE {
    VALUES ?item {"""+values_str+"""}
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """
    results = return_sparql_query_results(query_string)
    for result_i, result in enumerate(results['results']['bindings']):
        valuesD[final_values[result_i]] = result['itemLabel']['value']

