import datasets
from transformers import BertTokenizer, DistilBertModel,BertModel,DistilBertTokenizerFast
import pickle as pk
import torch
from collections import defaultdict

def prepare_chunked_sentences(tokenizer, text):
        lm_max_length = 510
        max_tokens = lm_max_length
        # if has_sos_eos:
            # max_tokens -= 2
        lm_chunk_length = max_tokens // 2
        if not hasattr(prepare_chunked_sentences, "sos_id"):
            prepare_chunked_sentences.sos_id, prepare_chunked_sentences.eos_id = tokenizer.encode("", add_special_tokens=True)
        tokenized_output = tokenizer(text,add_special_tokens=False, return_offsets_mapping=True)
        input_ids = tokenized_output["input_ids"]
        offset_mapping = tokenized_output["offset_mapping"]

        words = [] # concatenates wordpieces, there can also be other options such as removing subpieces
        words_to_input_ids = []
        for input_id, (char_start, char_end) in zip(input_ids, offset_mapping):
            word = text[char_start: char_end]
            decoded_word = tokenizer.convert_ids_to_tokens(input_id)
            
            if decoded_word != word: # wordpiece or if the starting character of a word is weird(edge case)
                # print(decoded_word, word) # for testing
                # assert decoded_word.startswith("##") # for testing, BERT
                if not words:
                    words.append("")
                    words_to_input_ids.append(list())
                
                words[-1] = words[-1] + word
                words_to_input_ids[-1].append(input_id)

            else:
                words.append(word)
                words_to_input_ids.append([input_id])

        input_ids_batches = []
        words_to_batch_position = []  # the position of each word in input_ids_batches (batch_id, start_id, end_id), [start, end)
        input_ids_current_batch = []
        for input_ids_for_word in words_to_input_ids:
            if len(input_ids_current_batch) + len(input_ids_for_word) > lm_max_length:
                input_ids_batches.append([prepare_chunked_sentences.sos_id] + input_ids_current_batch + [prepare_chunked_sentences.eos_id])
                if lm_chunk_length > 0:
                    input_ids_current_batch = input_ids_current_batch[-lm_chunk_length:]
                else:
                    input_ids_current_batch = []
            words_to_batch_position.append((len(input_ids_batches),
                                            len(input_ids_current_batch),
                                            len(input_ids_current_batch) + len(input_ids_for_word)))
            input_ids_current_batch.extend(input_ids_for_word)
        if len(input_ids_current_batch) > 0:
            input_ids_batches.append([prepare_chunked_sentences.sos_id] + input_ids_current_batch + [prepare_chunked_sentences.eos_id])
        return {
            "input_ids_batches": input_ids_batches,
            "words_to_batch_position": words_to_batch_position,
            "words": words
        }




with open("/data/mengke/.cache/NYT-Small/dataset.txt","r") as f:
    data = f.readlines()
#data = datasets.load_dataset("wikitext","wikitext-103-v1")
#data = data['train']

keywords = ["politics","art","business","science","sport"]
sentences = {k:list() for k in keywords}
for dt in data:
    for word in keywords:
        if " " + word + " " in dt.lower():
            sentences[word].append(dt.lower())
'''
for i in range(100000):
    for word in keywords:
        if " " + word + " " in data[i]["text"].lower():
            sentences[word].append(data[i]["text"].lower())
'''
for word in keywords:
    print(word + ": ", len(sentences[word]))


# save the relevant wiki paragraphs for diagonostics
print("Saving paragraphs...")
with open("NYT-Small.pk","wb") as f:
    pk.dump(sentences,f)
print("Saved.")


# pass through BERT
REPRESENTATION_SIZE = 768
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
#model = BertModel.from_pretrained("bert-base-uncased")
repr_dict = {}
for word in keywords:

    cnt = 0
    outputs = []
    num_paragraphs = len(sentences[word])
    key_sentences = sentences[word] 
    for t in key_sentences:
        if cnt % 10 == 0:
            print("processing number " + str(cnt) + "/" + str(num_paragraphs) + " paragraph")        
        cnt += 1
        processed_sentence = prepare_chunked_sentences(tokenizer,t)
        keyword_index = 0
        keyword_index = processed_sentence["words"].index(word)
        batch_id, start_id, end_id = processed_sentence["words_to_batch_position"][keyword_index]
        batches = processed_sentence["input_ids_batches"]
        output_batches = []
        for batch in batches:
            output = model(torch.tensor([batch]).to(torch.int64))["last_hidden_state"][0]
            output_batches.append(output)
        outputs.extend(output_batches[batch_id][start_id:end_id])
        print("output size: ", len(outputs[0]))

    print("number of representation vectors for " + word + ": ", len(outputs))
    '''
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
        
        encoded_input = tokenizer(t,return_tensors="pt",return_offsets_mapping=True)
        print(encoded_input)
        try:
            output = model(**encoded_input)
        except:
            continue
        for k in desired_output.keys():
            if word in k:
                for idx in desired_output[k]:
                    outputs.append(output['last_hidden_state'][0][idx])  
    '''
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
#print(repr_dict.keys())

with open("class_representation_temp.pk", "wb") as f:
    pk.dump(repr_dict,f)
print("Saved.")













