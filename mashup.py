# coding: utf-8

import nltk
from collections import Counter, defaultdict
import math
from gensim.models import word2vec
import re
import json
import click
import numpy as np

w2v = None
w2vbackup = None
def load_w2v():
    global w2v
    global w2vbackup
    print("Loading backup w2v")
    w2vbackup = word2vec.Word2Vec.load("data/gutenberg.w2v").similarity
    # print("Loading full w2v")
    # vecmat = np.load("data/google.npy")
    # vecmat_dict = json.load(open("data/google.json"))
    # def similarity(w1, w2):
    #     v1 = vecmat[vecmat_dict[w1],:] / 256.
    #     v2 = vecmat[vecmat_dict[w2],:] / 256.
    #     return np.dot(v1, v2)

    # print("Loaded")
    w2v = w2vbackup

stopwords = None
def load_stopwords():
    global stopwords
    stopwords = set(nltk.corpus.stopwords.words("english"))

    stopwords.update(["something", "nothing", "like", "could", "make", "made",
        "would", "isn", "hasn", "would", "shall", "one", "nobody", "somebody",
        "upon", "didn", "wouldn", "must", "aren", "couldn", "know", "maybe",
        "every", "even", "well", "said", "back", "says", "doesn", "go", "goes",
        "went", "going", "gone", "thing", "somewhere", "wasn", "might", "sir",
        "mrs", "miss", "shan", "can", "hadn", "won", "haven", "needn", "mayn",
        "weren", "shouldn", "get", "gets", "got", "getting", "let", "lets",
        "letting", "see", "sees", "seen", "saw", "seeing", "come", "coming",
        "came", "comes"])

def text(filename):
    return open(filename).read()

def wc(filename):
    if stopwords is None:
        load_stopwords()
    print("Computing frequencies for", filename)
    txt = text(filename)
    words = nltk.wordpunct_tokenize(txt)
    words = [w for w in words if w.isalpha()]
    count = Counter(words)
    bycaps = defaultdict(list)
    for w, c in count.items():
        bycaps[w.lower()].append((c, w))
    collapsed = {}
    total = 0
    for w, lst in bycaps.items():
        canon = max(lst)[1]
        if len(w) < 3 or w in stopwords:
            continue
        n = sum(x[0] for x in lst)
        collapsed[w] = (n, canon)
        total += n
    return collapsed, total


def pos_tag(word, tags={}):
    if not tags:
        try:
            print("Loading POS tags")
            tags.update(json.load(open("data/pos.json")))
            print("Loaded")
        except IOError:
            print("Counting POS tags")
            tagcounts = defaultdict(Counter)
            bonuses = defaultdict(int)
            bonuses["VBG"] = 1
            for w, t in nltk.corpus.brown.tagged_words():
                if not w.isalpha(): continue
                t = t.split('-')[0]
                tagcounts[w.lower()][t] += 1 + bonuses[t]
            for w in tagcounts.keys():
                tags[w] = tagcounts[w].most_common(1)[0][0]
            json.dump(tags, open("data/pos.json", "w"))
            print("Counted")
    try:
        return tags[word.lower()]
    except KeyError:
        tag = nltk.pos_tag([word])[0][1]
        if tag == 'NNP': tag = 'NP'
        if tag == 'NNPS': tag = 'NPS'
        tags[word.lower()] = tag
        return tag

def semantic_sim(w1, w2):
    if w2v is None and w2vbackup is None:
        load_w2v()
    try:
        return w2v(w1,w2)
    except KeyError:
        pass
    try:
        return w2vbackup(w1,w2)
    except KeyError:
        pass
    return 0.0

def build_tags(*wcs):
    tags = {}
    for wc in wcs:
        for w in wc.keys():
            tags[w] = pos_tag(wc[w][1])
    return tags

def match(structure, vocab):
    swc, slen = wc(structure)
    vwc, vlen = wc(vocab)
    translate = {}
    sfs = {}
    vfs = {}
    print("Grouping POS")
    tags = build_tags(swc, vwc)
    for w in swc.keys():
        sfs[w] = math.log(swc[w][0]/slen)
    for w in vwc.keys():
        vfs[w] = math.log(vwc[w][0]/vlen)
    sbyf = sorted(list(sfs.keys()), key=lambda x: -sfs[x])
    vbyf = sorted(list(vfs.keys()), key=lambda x: -vfs[x])
    print("Matching vocabulary")
    maxsem = 2
    minsem = -2
    good_tags = ["NN", "NP", "NNS", "NPS", "VB", "VBD", "VBZ", "VBG", "VBN", "JJ"]
    penalties = defaultdict(float)
    for i, sw in enumerate(sbyf):
        if i % 1000 == 0:
            print(i)
        if tags[sw] not in good_tags: continue
        if sw in vfs:
            bestscore = minsem + (sfs[sw] - vfs[sw]) ** 2 + penalties[sw]
        else:
            bestscore = 1e9
        best = sw
        for vw in vbyf:
            if tags[sw] != tags[vw]: continue
            freqscore = (sfs[sw] - vfs[vw]) ** 2
            if freqscore + minsem > bestscore:
                if vfs[vw] < sfs[sw]: break
                continue
            semanticscore = -2 * semantic_sim(swc[sw][1], vwc[vw][1])
            score = freqscore + semanticscore + penalties[vw]
            if score < bestscore:
                best = vw
                bestscore = score
        penalties[best] += math.log(swc[sw][0])
        if best != sw:
            translate[sw] = best
        if i < 100:
            print(sw, best)
    return translate


# In[318]:

def translate(filename, trans):
    txt = text(filename)
    def repl(match):
        word = match.group(0)
        try:
            repword = trans[word.lower()]
        except KeyError:
            return word
        if word.isupper():
            repword = repword.upper()
        elif word[0].isupper():
            repword = repword.capitalize()
        else:
            repword = repword.lower()
        return repword# + "[%s]" % word
    regex = re.compile(r'\w+|[^\w\s]+')
    print("Translating")
    newtxt = regex.sub(repl, txt)
    return fix_articles(newtxt)


# In[319]:

def transmatch(structure, vocab):
    return translate(structure, match(structure, vocab))

def fix_articles(txt):
    def fixup(match):
        oldart = match.group(0)
        spacing = match.group(2)
        firstchar = match.group(3)
        if firstchar in 'aeiouAEIOU':
            article = 'an'
        else:
            article = 'a'
        if oldart[0].isupper():
            article = article.capitalize()
        return article + spacing + firstchar
    return re.sub(r'\b(a|an)(\s+)([a-z])', fixup, txt, flags=re.IGNORECASE)

@click.group()
def cli():
    pass


@cli.command()
@click.argument("structure")
@click.argument("vocab")
@click.option("-o", "--output", default="mashup.txt")
def mash(structure, vocab, output):
    with open(output, "w") as f:
        f.write(transmatch(structure, vocab))

@cli.command()
@click.argument("filename")
def count(filename):
    counts, _ = wc(filename)
    for c, w in sorted(list(counts.values()), reverse=True)[:50]:
        print(w, c)

@cli.command()
@click.argument("filename")
def gender(filename):
    txt = text(filename)
    counts, _ = wc(filename)
    tags = build_tags(counts)
    nps = [counts[w][1] for w in sorted(list(tags.keys()), key=lambda x:
        counts[x][0]) if tags[w] == 'NP' and counts[w][0] > 5]
    pronouns = {"he": "m", "she": "f", "it": "n", "him": "m", "her": "f",
            "his": "m", "its": "n"}
    prons = re.findall(r"\b(he|she|it|him|her|his|its)\b", txt, flags=re.IGNORECASE)
    pcounts = Counter([pronouns[p.lower()] for p in prons])
    for name in nps:
        matches = re.findall(r"\b%s\b.*?\b(he|she|it|him|her|his|its)\b" % name,
                             txt,
                             flags=re.IGNORECASE | re.DOTALL)
        ncounts = Counter([pronouns[m.lower()] for m in matches])
        print(name, [(p, ncounts[p]/pcounts[p]) for p in "mfn"])

if __name__ == '__main__':
    cli()
