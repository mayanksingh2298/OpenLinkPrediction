wordnetD = dict()
wordnet_definitions = '/home/keshav/kg-bert/data/wn18rr/wordnet-mlj12-definitions.txt'
for line in open(wordnet_definitions):
    wid, name, description = line.strip('\n').split('\t')
    name = name.replace('_', ' ').strip()
    wordnetD[wid] = name

e2text_orig = '/home/keshav/kg-bert/data/wn18rr/entity2text.txt.orig'
e2text_new = open('/home/keshav/kg-bert/data/wn18rr/entity2text.txt','w')
for l in open(e2text_orig):
    id_, long_desc = l.strip('\n').split('\t')
    e2text_new.write(id_+'\t'+wordnetD[id_]+' '+long_desct+'\n')
e2text_new.close()
