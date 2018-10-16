word_count = {}

V = 0
with open('./full_text_sentences_new.txt', 'r') as f:
    for line in f:
    	#g.write(line)
        words = line.strip().split()
        for word in words:
            if word in word_count:
            	word_count[word] += 1
            else:
                word_count[word] = 1
                V += 1

l = sorted(word_count, key=word_count.get, reverse=True)
l = l[:10000]

i = 0
with open('./full_text_sentences_new.txt', 'r') as f, open('./small_corpus', 'w+') as g:
    for line in f:
	i += 1
	print i
        words = line.strip().split()
	words = [word for word in words if word in l]
	if len(words) > 0:
	    g.write(' '.join(words) + '\n')    
