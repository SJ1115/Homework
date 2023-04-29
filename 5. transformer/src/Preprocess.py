import re
import pickle
### Since We will bring tokenizer from outsides, in here we only performs clear-string.

def clean_str_eng(string):
    k = string

    ### String Clearing for English.
    ## General string-cleaning
    string = string.lower()
    string = re.sub(r"[^\w()/\-:\.!?\'\"]", " ", string)
    string = re.sub(r"\'s", "s", string) 
    string = re.sub(r"\'ve", " have", string) 
    string = re.sub(r"won\'t", "will not", string) 
    string = re.sub(r"n\'t", " not", string) 
    string = re.sub(r"\'re", " re", string) 
    string = re.sub(r"\'d", " d", string) 
    string = re.sub(r"\'ll", " will", string)
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
    string = re.sub(r'N{1,}', ' ', string) # ' <Num> '
    
    string = "<Sos> " + string + " <Eos>"
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() + '\n'

def clean_str_gen(string):
    """
    String Clearing for general language(German/French).
    """
    ## General string-cleaning
    string = string.lower()
    string = re.sub(r"[^\w()/\-:\.!?\'\"]", " ", string)
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
    string = re.sub(r'((-)?\d{1,3}(.\d{3})*(\,\d+)?)', 'N', string)
    string = re.sub(r'N{1,}', ' ', string) # ' <Num> '
    
    string = "<Sos> " + string + " <Eos>"
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() + '\n'

def clean_all(dataset, directory, max_len = 1000000):
    # Dev
    lst_en = []; lst_non = []

    iterator = (pair for pair in dataset['validation']['translation'])
    for pair in iterator:
        for lan in pair:
            if lan == 'en':
                lst_en.append(clean_str_eng(pair[lan]))
            else:
                lst_non.append(clean_str_gen(pair[lan]))

    for lan in pair:
        with open(directory + f"val_{lan}.txt", 'w') as f:
            #for line in (lst_en if lan=='en' else lst_non):
            f.writelines(lst_en if lan=='en' else lst_non)

    # Test
    lst_en = []; lst_non = []

    iterator = (pair for pair in dataset['test']['translation'])
    for pair in iterator:
        for lan in pair:
            if lan == 'en':
                lst_en.append(clean_str_eng(pair[lan]))
            else:
                lst_non.append(clean_str_gen(pair[lan]))

    for lan in pair:
        with open(directory + f"test_{lan}.txt", 'w') as f:
            #for line in (lst_en if lan=='en' else lst_non):
            f.writelines(lst_en if lan=='en' else lst_non)

    # Train
    lst_en = []; lst_non = []
    f_ind = 0

    iterator = (pair for pair in dataset['train']['translation'])
    for pair in iterator:
        for lan in pair:
            if lan == 'en':
                lst_en.append(clean_str_eng(pair[lan]))
            else:
                lst_non.append(clean_str_gen(pair[lan]))
            
        if len(lst_en) == max_len:
            for lan in pair:
                with open(directory + f"train_{lan}_{f_ind}.txt", 'w') as f:
                    #for line in (lst_en if lan=='en' else lst_non):
                    f.writelines(lst_en if lan=='en' else lst_non)
            lst_en = []
            lst_non = []
            f_ind += 1
