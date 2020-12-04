# coding: utf-8

import gensim
import math
from copy import copy

'''
(f) helper class, do not modify.
provides an iterator over sentences in the provided BNC corpus
input: corpus path to the BNC corpus
input: n, number of sentences to retrieve (optional, standard -1: all)
'''

class BncSentences:
	def __init__(self, corpus, n=-1):
		self.corpus = corpus
		self.n = n
	
	def __iter__(self):
		n = self.n
		ret = []
		for line in open(self.corpus):
			line = line.strip().lower()
			if line.startswith("<s "):
				ret = []
			elif line.strip() == "</s>":
				if n > 0:
					n -= 1
				if n == 0:
					break
				yield copy(ret)
			else:
				parts = line.split("\t")
				if len(parts) == 3:
					word = parts[-1]
					idx = word.rfind("-")
					word, pos = word[:idx], word[idx+1:]
					if word in ['thus', 'late', 'often', 'only', 'usually', 'however', 'lately', 'absolutely', 'hardly', 'fairly', 'near', 'similarly', 'sooner', 'there', 'seriously', 'consequently', 'recently', 'across', 'softly', 'together', 'obviously', 'slightly', 'instantly', 'well', 'therefore', 'solely', 'intimately', 'correctly', 'roughly', 'truly', 'briefly', 'clearly', 'effectively', 'sometimes', 'everywhere', 'somewhat', 'behind', 'heavily', 'indeed', 'sufficiently', 'abruptly', 'narrowly', 'frequently', 'lightly', 'likewise', 'utterly', 'now', 'previously', 'barely', 'seemingly', 'along', 'equally', 'so', 'below', 'apart', 'rather', 'already', 'underneath', 'currently', 'here', 'quite', 'regularly', 'elsewhere', 'today', 'still', 'continuously', 'yet', 'virtually', 'of', 'exclusively', 'right', 'forward', 'properly', 'instead', 'this', 'immediately', 'nowadays', 'around', 'perfectly', 'reasonably', 'much', 'nevertheless', 'intently', 'forth', 'significantly', 'merely', 'repeatedly', 'soon', 'closely', 'shortly', 'accordingly', 'badly', 'formerly', 'alternatively', 'hard', 'hence', 'nearly', 'honestly', 'wholly', 'commonly', 'completely', 'perhaps', 'carefully', 'possibly', 'quietly', 'out', 'really', 'close', 'strongly', 'fiercely', 'strictly', 'jointly', 'earlier', 'round', 'as', 'definitely', 'purely', 'little', 'initially', 'ahead', 'occasionally', 'totally', 'severely', 'maybe', 'evidently', 'before', 'later', 'apparently', 'actually', 'onwards', 'almost', 'tightly', 'practically', 'extremely', 'just', 'accurately', 'entirely', 'faintly', 'away', 'since', 'genuinely', 'neatly', 'directly', 'potentially', 'presently', 'approximately', 'very', 'forwards', 'aside', 'that', 'hitherto', 'beforehand', 'fully', 'firmly', 'generally', 'altogether', 'gently', 'about', 'exceptionally', 'exactly', 'straight', 'on', 'off', 'ever', 'also', 'sharply', 'violently', 'undoubtedly', 'more', 'over', 'quickly', 'plainly', 'necessarily']:
						pos = "r"
					if pos == "j":
						pos = "a"
					ret.append(gensim.utils.any2unicode(word + "." + pos))

'''
(a) function load_corpus to read a corpus from disk
input: vocabFile containing vocabulary
input: contextFile containing word contexts
output: id2word mapping word IDs to words
output: word2id mapping words to word IDs
output: vectors for the corpus, as a list of sparse vectors
'''
def load_corpus(vocabFile, contextFile):
	id2word = {}
	word2id = {}
	vectors = []
	
	it = 0 # to make the word - id mappings
	
	with open(vocabFile, 'r') as vf: 
	  # create the dictionaries word2id & id2word with vocab file
	  for line in vf:
	    line = line.strip()
	    word2id[line] = it
	    id2word[it] = line
	    it = it+1
	
	with open(contextFile, 'r') as cf:
	  # create the vector list with the contextFile
	  for line in cf:
	    temp_list = []
	    line = line.strip()
	    context_items = line.split(" ") 
	    for item in context_items[1:]: # Not take the first element, it is a number
	      item_elements = item.split(":")
	      temp_list.append((int(item_elements[0]), int(item_elements[1])))
	    vectors.append(temp_list)
	return id2word, word2id, vectors

'''
(b) function cosine_similarity to calculate similarity between 2 vectors
input: vector1
input: vector2
output: cosine similarity between vector1 and vector2 as a real number
'''
def cosine_similarity(vector1, vector2):
  # This cosine function internally use full vectors, but it
  # receives both full and sparse vectors
  temp_vec1 = [] # list that contains the full vector 1
  temp_vec2 = [] # list that contains the full vector 2
  
  # max size -----------------------------------------------
  if type(vector1[0]) is not tuple:
    N = len(vector1)
  elif type(vector2[0]) is not tuple:
    N = len(vector2)
  else:
    maxim = 0
    for elem in vector1:
      if elem[0]>maxim:
        maxim = elem[0]
    for elem in vector2:
      if elem[0]>maxim:
        maxim = elem[0]
    N = maxim+1
    
  # Transform sparse vectors to full vectors ---------------
  if type(vector1[0]) is tuple: # if the received vector is in sparse format:
    temp_vec1 = [0]*N # size of the vector space, 5000 context words
    for elem in vector1:
      temp_vec1[elem[0]] = elem[1]
  else:                 # else, it is already in full format
    temp_vec1 = vector1

  if type(vector2[0]) is tuple: # if the received vector is sparse:
    temp_vec2 = [0]*N # size of the vector space, 5000 context words
    for elem in vector2:
      temp_vec2[elem[0]] = elem[1]
  else:                 # else, it is already in full format
    temp_vec2 = vector2

  # performe the cosine= dotproduct(a,b)/(norm(a)*norm(b))
  dotproduct = 0
  denom1 = 0
  denom2 = 0
  for entry in range(0, len(temp_vec1)): # calculate element to element
    dotproduct = dotproduct + temp_vec1[entry]*temp_vec2[entry]
    denom1 = denom1 + temp_vec1[entry]*temp_vec1[entry]
    denom2 = denom2 + temp_vec2[entry]*temp_vec2[entry]

  cossim = dotproduct/(math.sqrt(denom1)*math.sqrt(denom2))
  return cossim

'''
(d) function tf_idf to turn existing frequency-based vector model into tf-idf-based vector model
input: freqVectors, a list of frequency-based vectors
output: tfIdfVectors, a list of tf-idf-based vectors
 '''
def tf_idf(freqVectors):
  # This converss the vector space into tf-idf space
  # Internally works with sparse vectors, but
  # receive both full and sparse formats

  tfIdfVectors = [] # vectors in tf-idf space
  freqVectors_sp = []
  N = len(freqVectors) # Total number of documents

  # Transform all the input into sparse vectors
  for col in freqVectors: # for each element in the list freqVectors
    temp = [] # temporary list that will contain the sparce vector
    if len(col) > 0 and (type(col[0]) is not tuple): # if it is a full vector and contains some element
      for item in range(0, len(col)): # iterate in this vector
        if col[item] > 0:
          temp.append((item, col[item])) # and make it sparse
      freqVectors_sp.append(temp)
    else: # if is an sparse vector do nothing
      freqVectors_sp.append(col)
      
  max_temp = 0
  for col in freqVectors_sp:
    for item in col:
      if(item[0]>max_temp):
        max_temp = item[0]
  df_vector = [0]*(max_temp+1) # (dfi) vector

  # Counts the number of documents that contains term i
  for col in freqVectors_sp:
    for item in col:
      df_vector[item[0]] += 1

  # Calculate de tf-idf for each element in freqVectors
  for col in freqVectors_sp:
    temp = []
    for item in col:
      idf_comp = (1+math.log(item[1],2))*(1+math.log((float(N)/df_vector[item[0]]),2))
      temp.append((item[0], idf_comp))
    tfIdfVectors.append(temp)

  return tfIdfVectors


'''
(f) function word2vec to build a word2vec vector model with 100 dimensions and window size 5
'''
def word2vec(corpus, learningRate, downsampleRate, negSampling):
	model = gensim.models.Word2Vec(corpus, size = 100, window = 5,
	               alpha = learningRate, sample = downsampleRate,
	               negative = negSampling)

	return model

'''
(h) function lda to build an LDA model with 100 topics from a frequency vector space
input: vectors
input: wordMapping mapping from word IDs to words
output: an LDA topic model with 100 topics, using the frequency vectors
'''

def lda(vectors, wordMapping):
  # Transform all the input into sparse vectors
  freqVectors_sp = []
  for col in vectors: # for each element in the list freqVectors
    temp = [] # temporary list that will contain the sparse vector
    if len(col) > 0 and (type(col[0]) is not tuple): # if it is a full vector
      for item in range(0, len(col)): # iterate in this vector
        if col[item] > 0:
          temp.append((item, col[item])) # and make it sparse
      freqVectors_sp.append(temp)
    else: # if is an sparse vector do nothing
      freqVectors_sp.append(col)
  model = gensim.models.LdaModel(corpus = freqVectors_sp,
	        id2word = wordMapping, update_every=0,
          passes = 10, num_topics = 100)
  return model

'''
(j) function get_topic_words, to get words in a given LDA topic
input: ldaModel, pre-trained Gensim LDA model
input: topicID, ID of the topic for which to get topic words
input: wordMapping, mapping from words to IDs (optional)
'''
def get_topic_words(ldaModel, topicID):
	words = ldaModel.show_topic(topicid = topicID, topn = 20)
	return words

if __name__ == '__main__':
	import sys

	part = sys.argv[1].lower()

	# these are indices for house, home and time in the data. Don't change.
	house_noun = 80
	home_noun = 143
	time_noun = 12

	# this can give you an indication whether part a (loading a corpus) works.
	# not guaranteed that everything works.
	if part == "a":
		print("(a): load corpus")
		try:
			id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
			if not id2word:
				print("\tError: id2word is None or empty")
				exit()
			if not word2id:
				print("\tError: id2word is None or empty")
				exit()
			if not vectors:
				print("\tError: id2word is None or empty")
				exit()
			print("\tPass: load corpus from file")
		except Exception as e:
			print("\tError: could not load corpus from disk")
			print(e)

		try:
			if not id2word[house_noun] == "house.n" or not id2word[home_noun] == "home.n" or not id2word[time_noun] == "time.n":
				print("\tError: id2word fails to retrive correct words for ids")
			else:
				print("\tPass: id2word")
		except Exception:
			print("\tError: Exception in id2word")
			print(e)

		try:
			if not word2id["house.n"] == house_noun or not word2id["home.n"] == home_noun or not word2id["time.n"] == time_noun:
				print("\tError: word2id fails to retrive correct ids for words")
			else:
				print("\tPass: word2id")
		except Exception:
			print("\tError: Exception in word2id")
			print(e)

	# this can give you an indication whether part b (cosine similarity) works.
	# these are very simple dummy vectors, no guarantee it works for our actual vectors.
	if part == "b":
		import numpy
		print("(b): cosine similarity")
		try:
			cos = cosine_similarity([(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)])
			if not numpy.isclose(0.5, cos):
				print("\tError: sparse expected similarity is 0.5, was {0}".format(cos))
			else:
				print("\tPass: sparse vector similarity")
		except Exception:
			print("\tError: failed for sparse vector")
		try:
			cos = cosine_similarity([1, 0, 1, 0, 2], [1, 2, 0, 0, 1])
			if not numpy.isclose(0.5, cos):
				print("\tError: full expected similarity is 0.5, was {0}".format(cos))
			else:
				print("\tPass: full vector similarity")
		except Exception:
			print("\tError: failed for full vector")

	# you may complete this part to get answers for part c (similarity in frequency space)
	if part == "c":
	  print("(c) similarity of house, home and time in frequency space")
	  id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
	  print("house-home:   " + str(cosine_similarity(vectors[word2id["home.n"]], vectors[word2id["house.n"]]))) 
	  print("home-time:   " + str(cosine_similarity(vectors[word2id["home.n"]], vectors[word2id["time.n"]])))
	  print("house-time:   " + str(cosine_similarity(vectors[word2id["house.n"]], vectors[word2id["time.n"]])))

	# this gives you an indication whether your conversion into tf-idf space works.
	# this does not test for vector values in tf-idf space, hence can't tell you whether tf-idf has been implemented correctly
	if part == "d":
		print("(d) converting to tf-idf space")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		try:
			tfIdfSpace = tf_idf(vectors)
			if not len(vectors) == len(tfIdfSpace):
				print("\tError: tf-idf space does not correspond to original vector space")
			else:
				print("\tPass: converted to tf-idf space")
		except Exception as e:
			print("\tError: could not convert to tf-idf space")
			print(e)

	# you may complete this part to get answers for part e (similarity in tf-idf space)
	if part == "e":
		print("(e) similarity of house, home and time in tf-idf space")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		vectors_tfidf = tf_idf(vectors)
		print("house-home:   " + str(cosine_similarity(vectors_tfidf[word2id["house.n"]], vectors_tfidf[word2id["home.n"]])))
		print("home-time:   " + str(cosine_similarity(vectors_tfidf[word2id["home.n"]], vectors_tfidf[word2id["time.n"]])))
		print("house-time:   " + str(cosine_similarity(vectors_tfidf[word2id["house.n"]], vectors_tfidf[word2id["time.n"]])))

	# you may complete this part for the first part of f (estimating best learning rate, sample rate and negative samplings)
	if part == "f1":# .05, .001, 10
	  print("(f1) word2vec, estimating best learning rate, sample rate, negative sampling")
	  learningRate_list = [0.01, 0.03, 0.05] 
	  downsampleRate_list = [0.0, 0.01, 0.001, 0.00001]
	  negSampling_list = [0, 5, 10]
	  model_acc = [0]*36
	  it = 0
	  for i in learningRate_list:
	    for j in downsampleRate_list:
	      for k in negSampling_list:
	        model = word2vec(BncSentences(corpus = "data/bnc.vert", n = 50000),
	        learningRate = i,
	        downsampleRate = j,
	        negSampling = k)
	        model_ac = model.accuracy("data/accuracy_test.txt")
	        correct = len(model_ac[14]["correct"])
	        incorrect = len(model_ac[14]["incorrect"])
	        model_acc[it] = float(correct)/(correct+incorrect)
	        print("modelo " + str(it) + "   accuracy: " + str(model_acc[it]))
	        it += 1

	# you may complete this part for the second part of f (training and saving the actual word2vec model)
	if part == "f2":
		import logging
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		print("(f2) word2vec, building full model with best parameters. May take a while.")
		final_model = word2vec(BncSentences(corpus = "data/bnc.vert"),
		learningRate = .05,
		downsampleRate = .001,
		negSampling = 10)
		final_model.save("modelo.word2vec")

	# you may complete this part to get answers for part g (similarity in your word2vec model)
	if part == "g":
		print("(g): word2vec based similarity")
		final_model = gensim.models.Word2Vec.load("modelo.word2vec")
		print("home-house:   "+ str(cosine_similarity(list(final_model["home.n"]),list(final_model["house.n"]))))
		print("home-time:   "+str(cosine_similarity(list(final_model["home.n"]),list(final_model["time.n"]))))
		print("house-time:   "+ str(cosine_similarity(list(final_model["house.n"]),list(final_model["time.n"]))))

	# you may complete this for part h (training and saving the LDA model)
	if part == "h":
		import logging
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		print("(h) LDA model")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		model_lda = lda(vectors, id2word)
		model_lda.save("modelo.lda")
		
	# you may complete this part to get answers for part i (similarity in your LDA model)
	if part == "i":
	  print("(i): lda-based similarity")
	  id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
	  model_lda = gensim.models.LdaModel.load("modelo.lda")
	  print(cosine_similarity(model_lda.get_document_topics(vectors[word2id["home.n"]],0),
	  model_lda.get_document_topics(vectors[word2id["house.n"]],0)))
	  print(cosine_similarity(model_lda.get_document_topics(vectors[word2id["home.n"]],0),
	  model_lda.get_document_topics(vectors[word2id["time.n"]],0)))
	  print(cosine_similarity(model_lda.get_document_topics(vectors[word2id["house.n"]],0),
	  model_lda.get_document_topics(vectors[word2id["time.n"]],0)))
	  
	

	# you may complete this part to get answers for part j (topic words in your LDA model)
	if part == "j":
		print("(j) get topics from LDA model")
		model_lda = gensim.models.LdaModel.load("modelo.lda")
		print(get_topic_words(model_lda, 80))
		print(get_topic_words(model_lda, 20))
		print(get_topic_words(model_lda, 50))
		print(get_topic_words(model_lda, 70))


