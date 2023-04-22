import urllib


class url:
    class train:
        en = "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en"
        de = "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de"
    class dev:
        en = "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en"
        de = "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de"
    class test:
        en = [
            "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en",
            "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.en"]
        de = [
            "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de",
            "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.de"]


def loader(name, train=True):
    enc = 'latin2' if train else 'utf-8'
    lines = urllib.request.urlopen(name).read().decode(enc)
    return lines.split("\n")