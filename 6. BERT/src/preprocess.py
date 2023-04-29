import re
import numpy as np
import collections
import json

from torch import tensor
from torch import long as tlong
from torch import int as tint
### Functional Tokens
PAD = "<Pad>"   # 0
MASK = "<Mask>" # 1
CLS = "<Cls>"   # 2
SEP = "<Sep>"   # 3
### Semantic Tokens
UNK = "<Unk>"   # 4
NUM = "<Num>"   # 5

####### Text Cleaning #######

def clean_str_book(string):
    ### String Clearing for HuggingFace_BookCorpus data.
    ## General string-cleaning
    string = string.lower()
    string = re.sub(r"[^a-zA-Z0-9()/\-:,\.!?\'\"`]", " ", string)
    
    string = re.sub(r"`", "'", string)
    string = re.sub(r"''", "\"", string)
    string = re.sub(r"\.\.\.", ".", string)

    string = re.sub(r"\'s", "\'s", string) 
    string = re.sub(r"\'ve", " have", string) 
    string = re.sub(r"won\'t", "will not", string) 
    string = re.sub(r"wo n\'t", "will not", string)
    string = re.sub(r"can\'t", "can not", string)
    string = re.sub(r"ca n\'t", "can not", string)
    string = re.sub(r"n\'t", " not", string) 
    string = re.sub(r"\'re", " re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"\'m", " am", string)
    
    string = re.sub(r"\'", " \' ", string)
    string = re.sub(r"\"", " \" ", string)
    string = re.sub(r"\-", " - ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"/", " / ", string)
    string = re.sub(r":", " : ", string)
    ## unify number to 'Num' toke
    string = re.sub(r'((-)?\d{1,3}(,\d{3})*(\.\d+)?)', 'N', string)
    string = re.sub(r'N{1,}', ' <Num> ', string) # ''
    
    #string = "<Sos> " + string + " <Eos>"
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() + "\n"

def clean_str_wiki(document):
    ### String Clearing for HuggingFace_wikipedia data.
    ## General string-cleaning
    doc = document.lower() ## a 'string' is a 'Document'
    
    chap = doc.split("\n") ## divided into 'Chapter'
    
    flag = True
    out = []
    for string in chap:
        if string.strip() == "see also":
            break  ## FootNotes after this keyword
        elif len(string.split()) < 7 :
            ## Chapter 'Title'. 7 is selected mannually.
            flag = False
            continue
        elif string[0] == "!":
            continue ## format line
        
        if not flag:
            out.append("\n")
        
        string = re.sub(r"[^a-zA-Z0-9()/\-:,\.!?\'\"`]", " ", string)
        
        string = re.sub(r"`", "'", string) 
        string = re.sub(r"''", "\"", string) 
        string = re.sub(r"\.\.\.", ".", string) 
        
        string = re.sub(r"\'ve", " have", string) 
        string = re.sub(r"won\'t", "will not", string) 
        string = re.sub(r"wo n\'t", "will not", string) 
        string = re.sub(r"can\'t", "can not", string) 
        string = re.sub(r"ca n\'t", "can not", string) 
        string = re.sub(r"n\'t", " not", string) 
        string = re.sub(r"\'re", " re", string) 
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " will", string)
        string = re.sub(r"\'m", " am", string)
        
        string = re.sub(r"\'", " \' ", string)
        string = re.sub(r"\' s ", " \'s ", string)
        string = re.sub(r"\"", " \" ", string)
        string = re.sub(r"\-", " - ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\.", " .", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"/", " / ", string)
        string = re.sub(r":", " : ", string)

        ## make line(sentence) segment
        string = re.sub(r'(\w{2,}) \. (\w)', r'\1 .\n\2', string)

        ## check abbreviation
        string = re.sub(r"dr \.\n", "dr ", string)
        string = re.sub(r"mr \.\n", "mr ", string)
        string = re.sub(r"mrs \.\n", "mrs ", string)
        string = re.sub(r"ms \.\n", "ms ", string)
        string = re.sub(r"inc \.\n", "inc ", string)
        string = re.sub(r"st \.\n", "st ", string)
        string = re.sub(r"cf \.\n", "cf ", string)
        

        ## unify number to 'Num' toke
        string = re.sub(r'((-)?\d{1,3}(,\d{3})*(\.\d+)?)', 'N', string)
        string = re.sub(r'N{1,}', ' <Num> ', string) # ''
        
        #string = "<Sos> " + string + " <Eos>"
        string = re.sub(r"\s{2,}", " ", string)

        string = string.strip() + "\n"

        out.append(string)
        # flag that the sentence just before was Real 
        flag = True
    return out

####### Transform to Masked-LM format #######

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [x for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [x for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if np.random.rand() < 0.5:
            trunc_tokens.pop(0)
        else:
            trunc_tokens.pop()

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, tokenizer):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == CLS or token == SEP:
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (len(cand_indexes) >= 1 and token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    np.random.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                        max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
            if is_any_index_covered:
                continue
        
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            r = np.random.random()
            if r < 0.8:
                masked_token = "<Mask>"
            # 10% of the time, keep original
            elif r < 0.9:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = tokenizer.id_to_token(np.random.randint(tokenizer.get_vocab_size()))

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)

def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, tokenizer):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for <Cls> and 2 <Sep>
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if np.random.random() < short_seq_prob:
        target_seq_length = np.random.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = np.random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or np.random.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = np.random.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = np.random.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append(CLS)
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append(SEP)
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append(SEP)
                segment_ids.append(1)

                (tokens, masked_lm_positions,
                masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, tokenizer)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances

def create_instances_from_lines(
        file_lines, out_file, min_term, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, tokenizer):
    """Creates `TrainingInstance`s for a single document."""
    lines = [tokenizer.encode(line).tokens for line in file_lines]

    # Account for <Cls> and 2 <Sep>
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    def set_seq_length(max_num_tokens, short_seq_prob):
        if np.random.random() < short_seq_prob:
            target_seq_length = np.random.randint(2, max_num_tokens)
        else:
            target_seq_length = max_num_tokens
        return target_seq_length
    
    
    target_seq_length = set_seq_length(max_num_tokens, short_seq_prob)
    
    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    
    start_ind = 0
    
    num_line = len(lines)
    
    #write
    #f = open(out_file, 'w')
    
    for i, line in enumerate(lines):
        current_chunk += line
        current_length += len(line)
        
        if (not line) or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) > 2:
                    a_end = np.random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.append(current_chunk[j])

                tokens_b = []
                is_random_next = 0
                
                # Random next
                if len(current_chunk) == 1 or np.random.random() < 0.5:
                    is_random_next = 1
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    while True:
                        random_line_index = np.random.randint(0, num_line - 1)
                        distance = max(start_ind - random_line_index,
                                       random_line_index - i)
                        
                        re = True
                        if distance > min_term:
                            while True:
                                random_line = lines[random_line_index]
                                if random_line:
                                    tokens_b.extend(random_line)
                                    if len(tokens_b) >= target_b_length:
                                        re = False
                                        break
                                else: # reset to empty document
                                    tokens_b = []
                                
                                random_line_index += 1
                                if random_line_index >= len(lines):
                                    break
                            
                            if re:
                                continue
                            else:
                                break
                    
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = 0
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.append(current_chunk[j])
                
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = [] # 0:padding, 1, 2
                tokens.append(CLS)
                segment_ids.append(0)
                
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append(SEP)
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append(SEP)
                segment_ids.append(1)


                (tokens, masked_lm_positions,
                masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, tokenizer)
                
                """instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)"""
                
                instances.append(
                    {"tokens" : tokenizer.encode(" ".join(tokens)).ids,
                     "segment_ids" : segment_ids,
                     "is_random_next" : is_random_next,
                     "masked_lm_positions" : masked_lm_positions,
                     "masked_lm_labels" : tokenizer.encode(" ".join(masked_lm_labels)).ids
                    }
                )
                
                #instances += 1
                #f.writelines(instance.__str__())
            
            target_seq_length = set_seq_length(max_num_tokens, short_seq_prob)
            current_chunk = []
            current_length = 0
            start_ind = i
    
    with open(out_file, "w") as f:
        json.dump(instances, f)
    #f.close()
    return len(instances)

def read_instance(batch, device='cpu'):
    """
    instance : dictionary with keys:
        "tokens", "masked_lm_labels", "masked_lm_positions"
        "segment_ids", "is_random_next"
    Out : 5 items with torch.tensor format
    """
    ## Input for model : like a batch 1
    tokens = tensor(batch['tokens']).unsqueeze(0).to(device)
    segment_ids = tensor(batch['segment_ids']).unsqueeze(0).to(device)
    ## Label for model
    masked_lm_labels = tensor(batch['masked_lm_labels']).unsqueeze(0).to(device)
    masked_lm_positions = tensor(batch['masked_lm_positions']).to(device)
    is_random_next = (batch['is_random_next']).to(device)
    
    return tokens, masked_lm_labels, masked_lm_positions, segment_ids, is_random_next

def custom_collate(batch):
    """
    instance : dictionary with keys:
        "tokens", "masked_lm_labels", "masked_lm_positions"
        "segment_ids", "is_random_next"
    Out : 5 items with torch.tensor format
    """
    ## Input for model : shape to batch 1 if needed
    tokens = tensor(batch[0]['tokens']).to(tint).unsqueeze(0)
    segment_ids = tensor(batch[0]['segment_ids']).to(tint).unsqueeze(0)
    masked_lm_positions = tensor(batch[0]['masked_lm_positions']).to(tint)
    ## Label for model
    masked_lm_labels = tensor(batch[0]['masked_lm_labels']).to(tlong).unsqueeze(0)
    is_random_next = tensor(batch[0]['is_random_next']).to(tlong).unsqueeze(0)
    
    return tokens, masked_lm_labels, masked_lm_positions, segment_ids, is_random_next