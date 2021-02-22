import spacy 
  
nlp = spacy.load('en_core_web_sm') 
    
sentence = "the south american youth championship" 
doc = nlp(sentence) 
import pdb
pdb.set_trace()
for ent in doc.ents: 
    print(ent.text, ent.start_char, ent.end_char, ent.label_) 
