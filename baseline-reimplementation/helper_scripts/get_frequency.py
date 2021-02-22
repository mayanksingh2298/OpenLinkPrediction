import sys
import os
sys.path.append(sys.path[0]+"/../")
import argparse
from kb import kb
import random
from tqdm import tqdm
import numpy as np
if __name__ == "__main__":
    # gets the frequency of each element
    ENTITY = 0 # must be 1 for entity frequncy otherwise 0 for relation frequency
    lis =  ['was released at',
 'is also spoken in',
 'serves children through',
 'made four appearances for',
 'is:impl_appos-clause home to',
 'has:impl_poss-clause experiences with',
 'is:impl_appos-clause southwest of',
 'is:impl_appos-clause also an enemy of',
 'is a village in the',
 'was the founder of',
 'is a novel by',
 'be:impl_vmod acquired by',
 'is an important component of',
 'was formed from',
 'is a public high school located in',
 'is:impl_appos-clause orton with',
 'was engulfed in',
 'be:impl_vmod shot by',
 'is the seventh studio album by',
 'be:impl_vmod granted to',
 'be:impl_vmod organized by',
 'is a romantic drama film directed by',
 'is:impl_appos-clause a member of',
 'is:impl_org-np-person of:impl_org-np-person driver',
 'is an american actress from',
 'leads to the production of',
 'had settled in',
 'performs the song in',
 'was an airline based in',
 'be:impl_vmod produced by',
 'be:impl_vmod run on',
 'is company town in',
 'was located in',
 'be investigated through',
 'created the role of',
 'played 135 games in',
 'is a fictional character portrayed by',
 'was introduced in',
 'is a skyscraper located in',
 'is home to',
 'is a commuter airline based in',
 'was a member of',
 'be:impl_vmod based in',
 'be:impl_vmod published in',
 'are some set of',
 'is:impl_org-np-person of:impl_org-np-person secretary-general',
 'is a science fiction novel by',
 'be:impl_vmod working for',
 'is:impl_appos-clause bruce shorts to',
 'is:impl_org-np-person of:impl_org-np-person admiral',
 'is a village in',
 'is:impl_appos-clause the president of',
 'is:impl_appos-clause then president of',
 'be:impl_vmod connecting to',
 'is a play by',
 'was traded to',
 'has:impl_poss-clause departments of',
 'is:impl_org-np-person of:impl_org-np-person architect',
 'is:impl_appos-clause asian part of',
 'lies approximately north of',
 'had its release in',
 'is:impl_appos-clause mayor of',
 'has been described by',
 'was president of',
 'be:impl_vmod held in',
 'took power under',
 'be:impl_vmod followed by',
 'is:impl_appos-clause puddle of',
 'is civil parish in',
 'be:impl_vmod established in',
 'has competed at',
 'was a norwegian politician for',
 'there are eight campuses within the cornwall college group at',
 'is:impl_appos-clause north of',
 'is:impl_org-np-person of:impl_org-np-person pitcher',
 'have played for',
 'be:impl_vmod spoken by',
 'is:impl_org-np-person of:impl_org-np-person president',
 'is a public high school in',
 'is:impl_appos-clause a disciple of',
 'is:impl_appos-clause in:impl_appos-clause canada near',
 'is made up of',
 'has:impl_poss-clause list of',
 'is:impl_appos-clause home ground of',
 'is:impl_appos-clause a variety of',
 'is:impl_org-np-person of:impl_org-np-person member',
 'helped establish the',
 'is:impl_appos-clause the region of',
 'be:impl_vmod situated in the',
 'competed at paralympics in',
 'is:impl_appos-clause point of',
 'making his league debut against',
 'be:impl_vmod held in',
 'is:impl_appos-clause a former member of',
 'became a member of',
 'is:impl_appos-clause president of',
 'became an independent commonwealth realm with',
 'warned national football league officials in',
 'came to be dominated by',
 'is:impl_appos-clause west of',]

    if ENTITY:
        path = "data/olpbench/mapped_to_ids/entity_id_map.txt"
    else:
        path = "data/olpbench/mapped_to_ids/relation_id_map.txt"
    mapp = {} # stores word to count freq
    lines = open(path,'r').readlines()
    for line in tqdm(lines[1:]): 
        line = line.strip().split("\t")
        mapp[line[0]] = int(line[2])

    ans = []
    for ele in lis:
        ans.append(mapp[ele])
        print(ans[-1])


