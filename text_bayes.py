from __future__ import division
import logging
import datetime
import math


class TextBayesClassifier(object):
    
    def __init__(self):
        self.vocabulary = {}
        self.examples = {}
        self.target_values = []
        self.target_value_probabilities = {}
    
    def load_example(self, example_name, tokens, target_value):
        """
        Read an example and update-initialize the corresponding frequencies for each token.
        """
        for token in tokens:
            
            try:
                self.vocabulary[token][target_value] += 1
            except KeyError:
                try:
                    self.vocabulary[token]
                except KeyError:
                    self.vocabulary[token] = {}
                
                try:
                    self.vocabulary[token][target_value]
                except KeyError:
                    self.vocabulary[token][target_value] = 1
            
            
        # TAKE CARE! IF THE SAME EXAMPLE COMES AGAIN WITH DIFFERENT TARGET VALUE THIS IS RENEWED!
        if(not example_name in self.examples.keys()):
            self.examples[example_name] = target_value
        elif(self.examples[example_name] != target_value):
            self.examples[example_name] = target_value
                
        if(target_value not in self.target_values):
            self.target_values += [target_value]
            
        #print example_name, "\t",delta_t,"\t", len(self.vocabulary)
                
    
    def guess(self, tokens):
        """
        Make a guess for a set of tokens, using the formula defined in the lecture's slides.
        But converting the production of probabilities to the sum of the respective logarithms.
        
        - postitions - all word positions in Doc that contain tokens in Vocabulary
        - Return v_NB, where:
          ....
        """
        argmax = None
        maxprob = None
        #notfound = {}
        
        for v_j in self.target_values:
           sigma = 0
           p_vj = math.log(self._get_P(v_j))
           
           for a_i in tokens:
               p_ai_vj = self._get_P(v_j, a_i)
               sigma += math.log(p_ai_vj[1])
           prob = sigma + p_vj 
          
          
           if prob > maxprob or maxprob == None:
               maxprob = prob
               argmax = v_j
        #for p in notfound:
        #    if len(notfound[p]) != 2:
        #        print "OUAOU" 
        return (argmax, maxprob)
                                     
                                        
    def train(self):
        """
        Calculate the required P(v_j) and P(w_k|v_j) probability terms
         
        For each target value vj in V do
        - docsj <- subset of Examples for which the target value is vj 
        - P(vj) <- |docsj| / |Examples|
        - Textj <- a single document created by concatenating all member of docsj 
        - n <- total number of words in Textj (counting duplicate words multiple times)
        - For each word w_k in Vocabulary
          - nk <- number of times word wk occurs in Textj
          - P(w_k|v_j) <- (n_k+1)/(n+|Vocabulary|)
          
        * Algorithm taken from the slides of the course: Machine learning - knowledge discovery by Stamatatos Efsathios (2010)
        """
        total_examples_length = len(self.examples.keys())
        for v_j in self.target_values:
            docs_j = filter(lambda example: self.examples[example] == v_j, self.examples)
            p_vj = len(docs_j) / total_examples_length
            n = self._get_total_number_of_words(target_value=v_j)
            self.target_value_probabilities[v_j] = {"probability": p_vj, "word_probabilities" : {}, "frequency":n}
            
            vocabulary_length = len(self.vocabulary.keys())
            
            # A bit obfuscated but provides better performance, the algorithm is described in the comments below
            #self.target_value_probabilities[v_j]["word_probabilities"] = dict(zip(self.vocabulary,[(self._get_total_occurencies_of(w_k, v_j) + 1)/ (n + vocabulary_length) for w_k in self.vocabulary]))

            
            for w_k in self.vocabulary:
                n_k = self._get_total_occurencies_of(w_k, v_j)
                p_w_k_v_j = (n_k + 1) / (n + vocabulary_length)
                #print w_k, n_k, p_vj, p_w_k_v_j, v_j
                self.target_value_probabilities[v_j]["word_probabilities"][w_k] = p_w_k_v_j
            
        #print self.target_value_probabilities
        #print self.vocabulary  
                

    def _get_total_number_of_words(self, target_value=None):
        """
        Returns the total number of words that can be found in the documents of a specific target value.
        """
        #total = 0
        #for word in self.vocabulary.keys():
        #    total += self._get_total_occurencies_of(word, target_value)   
        total = sum(map(lambda x: self._get_total_occurencies_of(x, target_value), self.vocabulary))
        return total


    def _get_total_occurencies_of(self, word, target_value=None):
        """
        Returns the total occurencies of a word in the documents of a specific target value
        or if no target value is specified in all the documents of all target values.
        """
        #print word, self.vocabulary[word]
        if target_value != None:
            if target_value in self.vocabulary[word].keys():
                return self.vocabulary[word][target_value]
            else: 
                return 0
        
        total = 0
        for target_value in self.vocabulary[word]:
            total += self.vocabulary[word][target_value]
        return total
    
    
    def _get_P(self, target_value, word=None):
        """
        Returns the probability of a target value, or if a word is specified the probability: P(word|target_value)
        """
        target_value_probability = self.target_value_probabilities[target_value]
        if(word):
            try:
                output = (True,target_value_probability["word_probabilities"][word])
            except KeyError:
                # NOTICE ! WHAT TO DO?!?!
                #probability = 1/( target_value_probability["frequency"] + len(self.vocabulary.keys())) 
                output = (False,0.00001)
            return output
        else:
            return target_value_probability["probability"]
            

   