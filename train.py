from data import *
from vectorized import *
from model import *

from torch.utils.tensorboard import SummaryWriter

def nllloss(y_pred, target):
    loss = 0
    for p_list, t_list in zip(y_pred, target):
        for p, t in zip(p_list, t_list):
            loss += -torch.log(p[t])
    return loss

def accuracy(y_pred, target):
    accur = torch.zeros(batch_size)
    for p_list, t_list in zip(y_pred, target):
        for i, (p, t) in enumerate(list(zip(p_list, t_list))):
            if torch.argmax(p) == t:
                accur[i] += 1

    return (accur == 2).sum()

class DataLoader:
    def __init__(self, data_set, batch_size):
        self.data_set = data_set
        self.batch_size = batch_size

    def ind_shuffle(self):
        return torch.randperm(self.data_set.data_set_length).view(-1, self.batch_size)

    def batches(self):
        indices = self.ind_shuffle() # .permute
        story_batch = self.data_set.story_tensors[indices]
        question_batch = self.data_set.question_tensors[indices]
        answer_batch = self.data_set.answer_tensors[indices]
        b = [{"story" : s, "question" : q, "answer" : a} for (s, q, a) in zip(story_batch, question_batch, answer_batch)]
        return b

def train_epoch(data_loader, model, optimizer):
    model.train()
    for batch in data_loader.batches():
        y_pred = model(batch["story"], batch["question"], batch["answer"])
        loss = nllloss(y_pred, batch["answer"].transpose(0, 1))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_epoch(data_loader, model):
    model.eval()
    loss = 0
    accur = 0
    with torch.no_grad():
        for batch in data_loader.batches():
            y_pred = model(batch["story"], batch["question"], batch["answer"])
            loss += nllloss(y_pred, batch["answer"].transpose(0, 1))
            accur += accuracy(y_pred, batch["answer"].transpose(0, 1))
            
    return loss, accur

voc, triplets = loadData(paths, corpus_qa1)
writer = SummaryWriter()

batch_size = 50
num_epoch = 400
hidden_size = 200  
embedding_size = 100

dmn = DMN(hidden_size, embedding_size, batch_size, voc.nmb_words)
#dmn.load_state_dict(torch.load("weights_398_epoch.pt"))

optimizer = torch.optim.SGD(dmn.parameters(), lr=0.001)

train_set = DataSet(triplets["train"], voc)
train_loader = DataLoader(train_set, batch_size)

test_set = DataSet(triplets["test"], voc)
test_loader = DataLoader(test_set, batch_size)

for epoch in range(num_epoch):
    train_epoch(train_loader, dmn, optimizer)
    torch.save(dmn.state_dict(), "weights_{}_epoch.pt".format(epoch))
    
    train_loss, train_accur = test_epoch(train_loader, dmn)
    test_loss, test_accur = test_epoch(test_loader, dmn)
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_accur, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', test_accur, epoch)
