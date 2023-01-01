## This page contains Directory / Encoding information.

def w2v():
    return "data/GoogleNews-vectors-negative300.bin"

def enc(data):
    if data in ("trec", "mr", "subj"):
        return "latin1"
    return "utf-8"

def location(data, mode='train'):
    directory = "data/text/"

    if data == 'sst1':
        mode = "phrases." + mode if mode == 'train' else mode
        return directory + "stsa.fine."+ mode
    
    elif data == 'sst2':
        mode = "phrases." + mode if mode == 'train' else mode
        return directory + "stsa.binary."+ mode

    elif data == 'cr':
        return directory + "custrev.all"

    elif data == 'mr':
        return directory + "rt-polarity.all"

    elif data == 'trec':
        return directory + "TREC." + mode + ".all"

    elif data == 'mpqa':
        return directory + "mpqa.all"

    elif data == 'subj':
        return directory + "subj.all"

def mode(name):
    mode = ['train']
    if name in ('sst1', 'sst2', 'trec'):
        mode.append('test')
        if not name == 'trec':
            mode.append('dev') 

    return mode
