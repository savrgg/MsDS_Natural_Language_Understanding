# coding: utf-8

from question1 import *
import json

'''
helper class to load a thesaurus from disk
input: thesaurusFile, file on disk containing a thesaurus of substitution words for targets
output: the thesaurus, as a mapping from target words to lists of substitution words
'''
def load_thesaurus(thesaurusFile):
	thesaurus = {}
	with open(thesaurusFile) as inFile:
		for line in inFile.readlines():
			word, subs = line.strip().split("\t")
			thesaurus[word] = subs.split(" ")
	return thesaurus

'''
(a) function addition for adding 2 vectors
input: vector1
input: vector2
output: addVector, the resulting vector when adding vector1 and vector2
'''
def addition(vector1, vector2):
  # This addition function internally use full vectors, but it
  # receives both full and sparse vectors
  temp_vec1 = [] # list that contains the full vector 1
  temp_vec2 = [] # list that contains the full vector 2
  vec_add = []
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
  if type(vector1[0]) is tuple:
    temp_vec1 = [0]*N 
    for elem in vector1:
      temp_vec1[elem[0]] = elem[1]
  else:                 
    temp_vec1 = vector1
  
  if type(vector2[0]) is tuple:
    temp_vec2 = [0]*N 
    for elem in vector2:
      temp_vec2[elem[0]] = elem[1]
  else:                 
    temp_vec2 = vector2

  vec_add = [0]*N
  for entry in range(0, len(temp_vec1)):
    vec_add[entry] = temp_vec1[entry] + temp_vec2[entry]
  
  # return to sparse vector
  temp = []
  for item in range(0, len(vec_add)):
    if vec_add[item] > 0:
      temp.append((item, vec_add[item])) 
  return temp

'''
(a) function multiplication for multiplying 2 vectors
input: vector1
input: vector2
output: mulVector, the resulting vector when multiplying vector1 and vector2
'''
def multiplication(vector1, vector2):
	# This multiplication function internally use full vectors, but it
  # receives both full and sparse vectors
  temp_vec1 = [] # list that contains the full vector 1
  temp_vec2 = [] # list that contains the full vector 2
  vec_add = []
  
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
    N = maxim + 1
    

  # Transform sparse vectors to full vectors ---------------
  if type(vector1[0]) is tuple:
    temp_vec1 = [0]*N 
    for elem in vector1:
      temp_vec1[elem[0]] = elem[1]
  else:                 
    temp_vec1 = vector1
  
  if type(vector2[0]) is tuple:
    temp_vec2 = [0]*N 
    for elem in vector2:
      temp_vec2[elem[0]] = elem[1]
  else:                 
    temp_vec2 = vector2

  vec_add = [0]*N
  for entry in range(0, len(temp_vec1)):
    vec_add[entry] = temp_vec1[entry] * temp_vec2[entry]
  
  # return to sparse vector
  temp = []
  for item in range(0, len(vec_add)):
    if vec_add[item] > 0:
      temp.append((item, vec_add[item])) 
  return temp

'''
(d) function prob_z_given_w to get probability of LDA topic z, given target word w
input: ldaModel
input: topicID as an integer
input: wordVector in frequency space
output: probability of the topic with topicID in the ldaModel, given the wordVector
'''
def prob_z_given_w(ldaModel, topicID, wordVector):
  wv = ldaModel.get_document_topics(wordVector, 0)
  temp = 0
  for item in wv:
    if topicID == item[0]:
      temp = item[1]
  return temp

'''
(d) function prob_w_given_z to get probability of target word w, given LDA topic z
input: ldaModel
input: targetWord as a string
input: topicID as an integer
output: probability of the targetWord, given the topic with topicID in the ldaModel
'''
def prob_w_given_z(ldaModel, targetWord, topicID):
  wv = ldaModel.show_topic(topicID, 2000)
  temp = 0
  for item in wv:
    if targetWord == item[0]:
      temp = item[1]
  return temp

'''
(f) get the best substitution word in a given sentence, according to a given model (tf-idf, word2vec, LDA) and type (addition, multiplication, lda)
input: jsonSentence, a string in json format
input: thesaurus, mapping from target words to candidate substitution words
input: word2id, mapping from vocabulary words to word IDs
input: model, a vector space, Word2Vec or LDA model
input: frequency vectors, original frequency vectors (for querying LDA model)
input: csType, a string indicating the method of calculating context sensitive vectors: "addition", "multiplication", or "lda"
output: the best substitution word for the jsonSentence in the given model, using the given csType
'''

def best_substitute(jsonSentence, thesaurus, word2id, model, frequencyVectors, csType):
  if csType == "addition":
    csv = []
    line = json.loads(jsonSentence)
    line_list = line["sentence"].split(" ")
    context_numbers = range(int(line["target_position"])-5, int(line["target_position"])+6)
    N_list = range(0, len(line_list))
    
    maxim = -1
    maxim_word = ""
    str_return = ""
    target_word = line_list[int(line["target_position"])]
    for subs_word in thesaurus[target_word]:
      #print(subs_word + "=======================")
      score = 0
      #print("score " + str(score))
      if(subs_word in word2id.keys()):
        if(type(model) == list):
          vector_subs = model[word2id[subs_word]]
        else:
          vector_subs = list(model[subs_word])
        for item in context_numbers:
          if(item in N_list and item != int(line["target_position"])):
            context_word = line_list[item]
            if(type(model) == list):
              if(target_word in word2id.keys() and context_word in word2id.keys()):
                vector_target = model[word2id[target_word]]
                vector_context = model[word2id[context_word]]
                if(len(vector_target)>0 and len(vector_context)>0):
                  csv = addition(vector_target, vector_context)
                  score = score + cosine_similarity(vector_subs, csv)
                  if(score > maxim):
                    maxim = score
                    maxim_word = subs_word
            else:
              if(target_word in model.wv.index2word and context_word in model.wv.index2word):
                vector_target = list(model[target_word])
                vector_context = list(model[context_word])
                if(len(vector_target)>0 and len(vector_context)>0):
                  csv = addition(vector_target, vector_context)
                  score = score + cosine_similarity(vector_subs, csv)
                  if(score > maxim):
                    maxim = score
                    maxim_word = subs_word
                  
    if(len(maxim_word) > 0):
      maxim_word = maxim_word[0:(len(maxim_word)-2)]
    str_return = target_word +" "+ line["id"] + " :: " + maxim_word
    print(str_return)
  elif csType == "multiplication":
    csv = []
    line = json.loads(jsonSentence)
    line_list = line["sentence"].split(" ")
    context_numbers = range(int(line["target_position"])-5, int(line["target_position"])+6)
    N_list = range(0, len(line_list))
    
    maxim = -1
    maxim_word = ""
    str_return = ""
    target_word = line_list[int(line["target_position"])]
    for subs_word in thesaurus[target_word]:
      #print(subs_word + "=======================")
      score = 0
      #print("score " + str(score))
      if(subs_word in word2id.keys()):
        if(type(model) == list):
          vector_subs = model[word2id[subs_word]]
        else:
          vector_subs = list(model[subs_word])
        for item in context_numbers:
          if(item in N_list and item != int(line["target_position"])):
            context_word = line_list[item]
            if(type(model) == list):
              if(target_word in word2id.keys() and context_word in word2id.keys()):
                vector_target = model[word2id[target_word]]
                vector_context = model[word2id[context_word]]
                if(len(vector_target)>0 and len(vector_context)>0):
                  csv = multiplication(vector_target, vector_context)
                  if(len(csv)>0):
                    score = score + cosine_similarity(vector_subs, csv)
                    if(score > maxim):
                      maxim = score
                      maxim_word = subs_word
            else:
              if(target_word in model.wv.index2word and context_word in model.wv.index2word):
                vector_target = list(model[target_word])
                vector_context = list(model[context_word])
                if(len(vector_target)>0 and len(vector_context)>0):
                  csv = multiplication(vector_target, vector_context)
                  if(len(csv)>0):
                    score = score + cosine_similarity(vector_subs, csv)
                    if(score > maxim):
                      maxim = score
                      maxim_word = subs_word
                    
    if(len(maxim_word) > 0):
      maxim_word = maxim_word[0:(len(maxim_word)-2)]
    str_return = target_word +" "+ line["id"] + " :: " + maxim_word
    print(str_return)
  elif csType == "lda":
    csv = []
    line = json.loads(jsonSentence)
    line_list = line["sentence"].split(" ")
    context_numbers = range(int(line["target_position"])-5, int(line["target_position"])+6)
    N_list = range(0, len(line_list))
    
    maxim = -1
    maxim_word = ""
    str_return = ""
    target_word = line_list[int(line["target_position"])]
    target_word_v = vectors[word2id[target_word]]
    pz_w = [0]*100
    for topicID in range(0,100):
      pz_w[topicID] = prob_z_given_w(model, topicID, target_word_v)
    for item in context_numbers:
      if(item in N_list and item != int(line["target_position"])):
        context_word = line_list[item]
        if(target_word in model.id2word.values() and context_word in model.id2word.values()):
          csv = [0]*100
          for topicID in range(0,100):
            csv[topicID] = prob_w_given_z(model, context_word, topicID)*pz_w[topicID]
                
    for subs_word in thesaurus[target_word]:
      score = 0
      if(subs_word in model.id2word.values()):
        vector_subs = model.get_document_topics(vectors[word2id[subs_word]]) # lda vector of vector_subs
        if(len(csv)>0):
          score = score + cosine_similarity(vector_subs, csv)
          if(score > maxim):
            maxim = score
            maxim_word = subs_word
                    
    if(len(maxim_word) > 0):
      maxim_word = maxim_word[0:(len(maxim_word)-2)]
    str_return = target_word +" "+ line["id"] + " :: " + maxim_word
    print(str_return)
    
  return str_return

if __name__ == "__main__":
	import sys

	part = sys.argv[1]

	# this can give you an indication whether part a (vector addition and multiplication) works.
	if part == "a":
		print("(a): vector addition and multiplication")
		v1, v2, v3 , v4 = [(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)], [1, 0, 1, 0, 2], [1, 2, 0, 0, 1]
		try:
			if not set(addition(v1, v2)) == set([(0, 2), (2, 1), (4, 3), (1, 2)]):
				print("\tError: sparse addition returned wrong result")
			else:
				print("\tPass: sparse addition")
		except Exception as e:
			print("\tError: exception raised in sparse addition")
			print(e)
		try:
			if not set(multiplication(v1, v2)) == set([(0,1), (4,2)]):
				print("\tError: sparse multiplication returned wrong result")
			else:
				print("\tPass: sparse multiplication")
		except Exception as e:
			print("\tError: exception raised in sparse multiplication")
			print(e)
		try:
			addition(v3,v4)
			print("\tPass: full addition")
		except Exception as e:
			print("\tError: exception raised in full addition")
			print(e)
		try:
			multiplication(v3,v4)
			print("\tPass: full multiplication")
		except Exception as e:
			print("\tError: exception raised in full addition")
			print(e)

	# you may complete this to get answers for part b (best substitution words with tf-idf and word2vec, using addition)
	if part == "b":
		print("(b) using addition to calculate best substitution words")
		vocabFile = "data/test.txt"
		thesaurus = load_thesaurus("data/test_thesaurus.txt")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		tf_vectors = tf_idf(vectors)
		word2vec_model = gensim.models.Word2Vec.load("modelo.word2vec")
		
		target = open("tf-idf_addition.txt", 'w')
		with open(vocabFile, 'r') as vf:
		  for line in vf:
		    line = best_substitute(line, thesaurus, word2id, tf_vectors, vectors, "addition")
		    target.write(line + "\n")
		    
		target = open("word2vec_addition.txt", 'w')
		with open(vocabFile, 'r') as vf: 
		  for line in vf:
		    line = best_substitute(line, thesaurus, word2id, word2vec_model, vectors, "addition")
		    target.write(line + "\n")
		    
	# you may complete this to get answers for part c (best substitution words with tf-idf and word2vec, using multiplication)
	if part == "c":
		print("(c) using multiplication to calculate best substitution words")
		vocabFile = "data/test.txt"
		thesaurus = load_thesaurus("data/test_thesaurus.txt")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		tf_vectors = tf_idf(vectors)
		word2vec_model = gensim.models.Word2Vec.load("modelo.word2vec")
		
		target = open("tf-idf_multiplication.txt", 'w')
		with open(vocabFile, 'r') as vf: 
		  for line in vf:
		    line = best_substitute(line, thesaurus, word2id, tf_vectors, vectors, "multiplication")
		    target.write(line + "\n")
		    
		target = open("word2vec_multiplication.txt", 'w')
		with open(vocabFile, 'r') as vf: 
		  for line in vf:
		    line = best_substitute(line, thesaurus, word2id, word2vec_model, vectors, "multiplication")
		    target.write(line + "\n")

	# this can give you an indication whether your part d1 (P(Z|w) and P(w|Z)) works
	if part == "d":
		print("(d): calculating P(Z|w) and P(w|Z)")
		print("\tloading corpus")
		id2word,word2id,vectors=load_corpus(sys.argv[2], sys.argv[3])
		print("\tloading LDA model")
		ldaModel = gensim.models.ldamodel.LdaModel.load("modelo.lda")
		houseTopic = ldaModel[vectors[word2id["house.n"]]][0][0]
		try:
			if prob_z_given_w(ldaModel, houseTopic, vectors[word2id["house.n"]]) > 0.0:
				print("\tPass: P(Z|w)")
			else:
				print("\tFail: P(Z|w)")
		except Exception as e:
			print("\tError: exception during P(Z|w)")
			print(e)
		try:
			if prob_w_given_z(ldaModel, "house.n", houseTopic) > 0.0:
				print("\tPass: P(w|Z)")
			else:
				print("\tFail: P(w|Z)")
		except Exception as e:
			print("\tError: exception during P(w|Z)")
			print(e)

	# you may complete this to get answers for part d2 (best substitution words with LDA)
	if part == "e":
		print("(e): using LDA to calculate best substitution words")
		vocabFile = "data/test.txt"
		thesaurus = load_thesaurus("data/test_thesaurus.txt")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		model_lda = gensim.models.LdaModel.load("modelo.lda")
		
		target = open("output_lda.txt", 'w')
		with open(vocabFile, 'r') as vf: 
		  for line in vf:
		    line = best_substitute(line, thesaurus, word2id, model_lda, vectors, "lda")
		    target.write(line + "\n")
