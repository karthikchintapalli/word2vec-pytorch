import torch
from torch.autograd import Variable
import math
import sys
import random

#dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor

word_count = {}
context_len = 2
unk = '<UNK>'

SEED = 5

V = 1
emb_dim = 100
learning_rate = 0.025
no_of_epochs = 15

with open('./small_corpus', 'r') as f:
    for line in f:
        words = line.strip().split()
        for word in words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
                V += 1

word_count[unk] = 0
words = [unk] + sorted(word_count, key=word_count.get, reverse=True)
index = {word : i for i, word in enumerate(words)}

z = [word_count[word] for word in words]
total = sum(z)
z = [x/total for x in z]

p = [(math.sqrt(x/0.001) + 1) * (0.001/x) if x != 0 else 0 for x in z]

data = []
labels = []

with open('./small_corpus', 'r') as f:
    for line in f:
        sent = line.strip().split()
        sent = [word for word in sent if random.random() <= p[index[word]]]
        sent = [unk for i in xrange(context_len)] + sent + [unk for i in xrange(context_len)]
        sent = [index[word] for word in sent]
        for i in xrange(context_len, len(sent) - context_len):
        	for j in xrange(i - context_len, i):
        		if sent[j] != 0:
        			data.append(sent[i])
        			labels.append(sent[j])

        	for j in xrange(i + 1, i + 1 + context_len):
        		if sent[j] != 0:
        			data.append(sent[i])
        			labels.append(sent[j])

random.seed(SEED)
random.shuffle(data)
random.seed(SEED)
random.shuffle(labels)

data = data[:1000]
labels = labels[:1000]

w1 = Variable(torch.randn(V, emb_dim).type(dtype), requires_grad=True)
w2 = Variable(torch.zeros(emb_dim, V).type(dtype), requires_grad=True)

for epoch in xrange(1, no_of_epochs + 1):
	print "epoch " + str(epoch)
	losses = []
	for i in xrange(len(data)):	
		sys.stdout.write("%f%%       \r" % (i * 100.0/len(data)))
		sys.stdout.flush()
		x = torch.zeros(1, V).type(dtype)
		x[0, data[i]] = 1
		x = Variable(x, requires_grad=False)

		h1 = x.mm(w1)
		
		softmax = torch.nn.Softmax(dim=1)
		output = softmax(h1.mm(w2))

		loss = -1.0 * torch.log(output[0, labels[i]])
		loss.backward()

		w1.data -= learning_rate * w1.grad.data
		w2.data -= learning_rate * w2.grad.data

		w1.grad.data.zero_()
		w2.grad.data.zero_()
		losses.append(loss.data[0])

	print ""
	print "loss: " + str(sum(losses) * 1.0/len(losses))

with open('./sg_vecs', 'w+') as f:
	f.write(','.join(words) + '\n')
	for i in xrange(V):
		f.write(','.join([str(x) for x in w1.data[i, :].numpy().tolist()]) + '\n')
