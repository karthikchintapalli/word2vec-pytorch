import numpy as np
from scipy.spatial.distance import cosine

def find_n_closest(word, n):
	if word in words:
		v = word_vecs[words.index(word), :]
		dists = []
		for i in xrange(len(words)):
			if words[i] != word:
				dists.append((i, 1.0 - cosine(v, word_vecs[i, :])))

		dists = sorted(dists, key=lambda x: x[1], reverse=True)
		dists = dists[:n]

		for dist in dists:
			print words[dist[0]], dist[1]

	else:
		print "word doesn't exist"

word_vecs = []
words = []
with open('./cbow_vecs', 'r') as f:
	lc = 0
	for line in f:
		lc += 1
		if lc == 1:
			words += line.strip().split(',')

		else:
			word_vecs.append([float(x) for x in line.strip().split(',')])

word_vecs = np.array(word_vecs)

while True:
	word = raw_input()
	find_n_closest(word, 10)
