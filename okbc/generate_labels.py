from tqdm import tqdm
inp_f = open('data/debug.txt','r')
out_f = open('data/debug_labels.txt','w')
e2_set = set()
for line in tqdm(inp_f):
    line = line.strip('\n')
    e2 = line.split('\t')[2]
    e2_set.add(e2)
for e2 in e2_set:
    out_f.write(e2+'\n')
#out_f.write('unknown\n')
out_f.close()
