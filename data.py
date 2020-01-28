import os
import unicodedata
import re
import random

# special tokens
PAD_token = 0
EOS_token = 1

# corpus names
corpus_qa1 = "qa1_single-supporting-fact"
corpus_qa2 = "qa2_two-supporting-facts"

# data paths
path_train = "data/" + corpus_qa2 + "_train.txt"
path_test = "data/" + corpus_qa2 + "_test.txt"
paths = {"train" : path_train, "test" : path_test}

class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", EOS_token: "EOS"}
        self.nmb_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.nmb_words
            self.word2count[word] = 1
            self.index2word[self.nmb_words] = word
            self.nmb_words += 1
        else:
            self.word2count[word] += 1
            
    def print_sentence(self, tokens):
        for tok in tokens:
            print(self.index2word[tok.item()], end=' ')
        print()
        
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", s)
    return s

def isQuestionLine(line):
    return len(line) == 3

def isNewStoryLine(id):
    return id == "1"

def tripletsFromLines(lines):
    triplets = []
    
    for line in lines:
        line_id, line_base = line[0].split(' ', 1) # line[0] ? star expr?
        
        if isNewStoryLine(line_id):
            story = [line_base]
        elif isQuestionLine(line):
            question = line_base
            answer = line[1]
            triplet = {"story" : story.copy(), "question" : question, "answer" : answer}
            triplets.append(triplet)
        else:
            story.append(line_base)

    return triplets
            
def readData(path):
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    lines = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    triplets = tripletsFromLines(lines)
    return triplets

def loadData(paths, corpus_name):
    print("Reading [" + corpus_name + "] data...")
    
    train_triplets = readData(paths["train"])
    test_triplets = readData(paths["test"])
    
    voc = Voc(corpus_name)

    triplets = train_triplets + test_triplets
    for triplet in triplets:
        inputs, question, answer = triplet.values()
        for sentence in inputs:
            voc.addSentence(sentence)
        voc.addSentence(question)
        voc.addSentence(answer)

    return voc, {"train" : train_triplets, "test" :  test_triplets}
