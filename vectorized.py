import torch
import itertools
from data import PAD_token, EOS_token

class DataSet:
    def __init__(self, triplets, voc):
        self.triplets = triplets
        self.voc = voc
        self.data_set_length = len(triplets)
        
        self.story_tensors, self.story_seq_lengths = self.padTensors(self.tensorFromStory, "story")
        self.question_tensors, self.question_seq_lengths = self.padTensors(self.tensorFromQuestion, "question")
        self.answer_tensors, self.answer_seq_lengths = self.padTensors(self.tensorFromAnswer, "answer")
            
    def indexesFromSentence(self, sentence):
        return [self.voc.word2index[word] for word in sentence.split(' ')]

    def tensorFromStory(self, story):
        indexes = [self.indexesFromSentence(sentence) + [EOS_token] for sentence in story]
        indexes = list(itertools.chain.from_iterable(indexes)) # rewrite
        return torch.LongTensor(indexes)

    def tensorFromQuestion(self, question):
        indexes = self.indexesFromSentence(question)
        return torch.LongTensor(indexes)
    
    def tensorFromAnswer(self, answer):
        indexes = self.indexesFromSentence(answer) + [EOS_token]
        return torch.LongTensor(indexes)
    
    def padTensors(self, tensorFromData, kind):
        seqs = [tensorFromData(triplet[kind]) for triplet in self.triplets]
        seq_lengths = torch.LongTensor([len(seq) for seq in seqs])
        tensors = torch.empty((self.data_set_length, seq_lengths.max())).fill_(PAD_token).long()
        for idx, (seq, seqlen) in enumerate(zip(seqs, seq_lengths)):
            tensors[idx, :seqlen] = seq

        return tensors, seq_lengths 
