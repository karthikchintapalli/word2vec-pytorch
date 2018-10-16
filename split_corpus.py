word_count = {}
vocab_size = 10000

V = 0
with open('./full_text_sentences_new.txt', 'r') as f, open('./small_corpus', 'w+') as g:
    for line in f:
    	g.write(line)
        words = line.strip().split()
        for word in words:
            if word in word_count:
            	continue
            else:
                word_count[word] = 1
                V += 1
        if V > vocab_size:
        	break