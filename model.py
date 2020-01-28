import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from data import PAD_token, EOS_token

TM = 4

class InputModule(nn.Module):
    def __init__(self, hidden_size, embedding_size, batch_size, weights):
        super(InputModule, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, input_tokens):
        hidden = self.initHidden()

        input_tokens = input_tokens.transpose(0, 1)
        embedded = self.embedding(input_tokens)
        
        outputs, _ = self.gru(embedded, hidden)

        mask = input_tokens == EOS_token
        c = torch.zeros(mask.sum(0).max(), self.batch_size, self.hidden_size)

        for i in range(self.batch_size):
            c[:, i][:len(outputs[:, i][mask[:, i]])] = outputs[:, i][mask[:, i]]

        return c

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size)

class QuestionModule(nn.Module):
    def __init__(self, hidden_size, embedding_size, batch_size, weights):
        super(QuestionModule, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, question_tokens):
        hidden = self.initHidden()

        question_tokens = question_tokens.transpose(0, 1)
        embedded = self.embedding(question_tokens)
        _, hidden = self.gru(embedded, hidden)

        return hidden.squeeze(0)

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size)
    
class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.ln1 = nn.Linear(7 * hidden_size, 7 * hidden_size)
        self.ln2 = nn.Linear(7 * hidden_size, 1)
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)

    def z(self, c, m, q):
        return torch.cat((c, m, q, c*q, c*m, torch.abs(c-q), torch.abs(c-m)), 1)
    
    def forward(self, c, m, q, h):
        zi = self.z(c, m, q)
        gate = torch.tanh(self.ln1(zi))
        gate = torch.sigmoid(self.ln2(gate))
        #print(gate, end=' ')
        hidden = gate*self.gru_cell(c, h) + (1 - gate)*h
        return hidden
    
class EpisodicMemoryModule(nn.Module):
    def __init__(self, hidden_size, batch_size):
        super(EpisodicMemoryModule, self).__init__()
        
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.attn = Attn(hidden_size)
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, fact_represent, q):
        m = q
        for i in range(TM):
            #print("Episode {}: ".format(i), end='')
            h = self.initHidden()
            for c in fact_represent:
                h = self.attn(c, m, q, h)
                
            m = self.gru_cell(h, m)
            #print('')
        return m

    def initHidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)

class AnswerModule(nn.Module):
    def __init__(self, hidden_size, embedding_size, batch_size, vocab_size):
        super(AnswerModule, self).__init__()
        
        self.hidden_size = hidden_size 
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        
        self.ln = nn.Linear(hidden_size, vocab_size, bias=False)
        self.gru_cell = nn.GRUCell(vocab_size + hidden_size, hidden_size)

    def forward(self, a, q, answer_tokens):
        y = torch.softmax(self.ln(a), 1) 

        outputs = torch.empty((2, self.batch_size, self.vocab_size))
        for i in range(2):
            a = self.gru_cell(torch.cat((y, q), 1), a)
            y = torch.softmax(self.ln(a), 1)
            outputs[i] = y

        return outputs

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

class DMN(nn.Module):
    def __init__(self, hidden_size, embedding_size, batch_size, vocab_size):
        super(DMN, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
         
        emb_weights = torch.randn(vocab_size, embedding_size)        
        self.input_module = InputModule(hidden_size, embedding_size, batch_size, emb_weights)
        self.question_module = QuestionModule(hidden_size, embedding_size, batch_size, emb_weights)
        self.episodic_memory_module = EpisodicMemoryModule(hidden_size, batch_size)
        self.answer_module = AnswerModule(hidden_size, embedding_size, batch_size, vocab_size)
        
    def forward(self, story, question, answer):
        fact_represent = self.input_module(story)
        q = self.question_module(question)
        m = self.episodic_memory_module(fact_represent, q)
        
        y_pred = self.answer_module(m, q, answer)
        return y_pred
