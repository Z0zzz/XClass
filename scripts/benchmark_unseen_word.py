import datasets
from transformers import BertTokenizer, BertModel
import pickle as pk
import torch
from collections import defaultdict

data = datasets.load_dataset("wikitext","wikitext-103-v1")
data = data['train']

keywords = ["politics","art","business","science","sport"]
sentences = {k:list() for k in keywords}
for i in range(10000):
    for word in keywords:
        if word in data[i]['text'].lower():
            sentences[word].append(data[i]['text'].lower())
for word in keywords:
    print(len(sentences[word]))

# save the relevant wiki paragraphs for diagonostics
print("Saving paragraphs...")
with open("NYT-Small_wikitext_10000.pk","wb") as f:
    pk.dump(sentences,f)
print("Saved.")

# pass through BERT
REPRESENTATION_SIZE = 768
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
repr_dict = {}
for word in keywords:

    cnt = 0
    outputs = []
    num_paragraphs = len(sentences[word])
    key_sentences = sentences[word] 

    # process and extract keyword static representation
    for t in key_sentences:
        if cnt % 5 == 0:
            print("processing number " + str(cnt) + "/" + str(num_paragraphs) + " paragraph")
        
        cnt += 1
        enc = {x:tokenizer.encode(x,add_special_tokens=False) for x in t.split()}
        desired_output = defaultdict(list)
        idx = 1
        
        for token,encoded in enc.items():
            token_output = []
            for ids in encoded:
                token_output.append(idx)
                idx += 1
            desired_output[token].extend(token_output)
        
        encoded_input = tokenizer(t,return_tensors="pt")
        try:
            output = model(**encoded_input)
        except:
            continue

        for idx in desired_output[word]:
            outputs.append(output['last_hidden_state'][0][idx])
    
    # average over the output to get the static word representation
    num_repres = 0
    repr_vector = torch.zeros(REPRESENTATION_SIZE)
    for t in outputs:
        repr_vector += t
        num_repres += 1
    repr_vector /= num_repres
    repr_dict[word] = repr_vector.detach().numpy()
print("Saving word vector representation...")
# save
print(repr_dict.keys())

with open("class_representation_NYT-Small.pk", "wb") as f:
    pk.dump(repr_dict,f)
print("Saved.")













