###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids:  adisoni-nsadhuva-svaddi
#
# (Based on skeleton code by D. Crandall)
#
import datetime
import json
import pprint
import random
import math
import copy
from threading import Thread
from threading import Lock


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
import numpy as np

class Word:
    def __init__(self, word, POS):
        self.word = word
        self.pos_dict = {"det": 0, "noun": 0, "pron": 0,
                         "adj": 0, "adv": 0, "adp": 0,
                         "num": 0, "conj": 0, "x": 0,
                         ".": 0, "prt": 0, "verb": 0}
        self.lock = Lock()
        self.POS = POS
        self.update_pos_count(POS)
        self.total_occurances = 1
        self.emission_calculated = False
        self.emission = copy.deepcopy(self.pos_dict)

    def max_pos(self):
        init = 0
        max_pos_ret = ""
        for i in self.emission:
            if self.emission[i] >= init:
                init = self.emission[i]
                max_pos_ret = i
        return max_pos_ret, init

    # decorator
    def _lock(func):
        def wrapper(self, *args, **kwargs):
            self.lock.acquire()
            r = func(self, *args, **kwargs)
            self.lock.release()
            return r
        return wrapper

    @_lock
    def update_pos_count(self, POS):
        """
        updates the count for respective POS for this word
        :return:
        """
        c = self.pos_dict[POS]
        self.pos_dict.update({POS: c + 1})

    @_lock
    def update_occurances(self):
        self.total_occurances += 1
    #
    # @_lock
    # def update_line_num(self, line_num):
    #     self.line_nums.append(line_num)

    def get_occurances(self):
        return self.total_occurances

    def get_pos_count(self, pos):
        return self.pos_dict[pos]

    def get_emission(self, pos):
        return self.emission[pos]

    def set_emission(self, emission, pos):
        self.emission.update({pos: emission})
        self.emission_calculated = True

class BagOfWords:
    def __init__(self):
        self.pos_dict = {"det": 0, "noun": 0, "pron": 0,
                         "adj": 0, "adv": 0, "adp": 0,
                         "num": 0, "conj": 0, "x": 0,
                         ".": 0, "prt": 0, "verb": 0}
        self.bag = {}
        self.lock = Lock()
        self.wc = 0

    # decorator
    def _lock(func):
        def wrapper(self, *args, **kwargs):
            self.lock.acquire()
            r = func(self, *args, **kwargs)
            self.lock.release()
            return r
        return wrapper

    @_lock
    def update_word(self, word, POS):
        self.wc += 1
        if word in self.bag:
            #   need to update the respective POS COUNT
            self.bag[word].update_pos_count(POS)
            self.bag[word].update_occurances()
        else:
            #   create word object and append to bag
            self.bag.update({word: Word(word, POS)})
        self.pos_dict.update({POS: self.pos_dict[POS] + 1})

        
    def emissions(self, word, pos):
        """
        P( Word| POS) = occurances where the word is that POS/ total occurances of that word
        :return:
        """
        P_Word_given_POS = self.bag[word].get_pos_count(pos) / self.bag[word].get_occurances()
        e = pow(10, -10)
        if 0 not in [P_Word_given_POS]:
           self.bag[word].set_emission(P_Word_given_POS, pos)
        else:
            self.bag[word].set_emission(e, pos)

#In order to calculate the initial and transition probabilities that will be required for the Viterbi Algorithm
#Makes a 13*13 Matrix with all the 12 parts of speech and initial probabilities
class TheMatrix:
    def __init__(self):
        self.pos_dict = {"det": 0, "noun": 0, "pron": 0,
                         "adj": 0, "adv": 0, "adp": 0,
                         "num": 0, "conj": 0, "x": 0,
                         ".": 0, "prt": 0, "verb": 0, "initial": 0}
        self.all_pos = {}
        for i in self.pos_dict:
            self.pos_dict[i]=1
        for i in self.pos_dict:
            self.all_pos.update({i: copy.deepcopy(self.pos_dict)})
        self.all_pos.pop("initial")
        self.pos_dict.pop("initial")
        self.all_probs = copy.deepcopy(self.all_pos)
        self.lock = Lock()

    # decorator
    def _lock(func):
        def wrapper(self, *args, **kwargs):
            self.lock.acquire()
            r = func(self, *args, **kwargs)
            self.lock.release()
            return r
        return wrapper

    @_lock
    def update(self, pos_list):
        self.all_pos[pos_list[0]]["initial"] += 1
        for i in range(len(pos_list) - 1):
            c = self.all_pos[pos_list[i]][pos_list[i + 1]]
            temp = self.all_pos[pos_list[i]]
            temp.update({pos_list[i + 1]: c + 1})
            self.all_pos.update({pos_list[i]: temp})

    def get_row_sum(self):
        self.row_sums = {}
        for pos in self.pos_dict:
            self.row_sums.update({pos : sum(list(self.all_pos[pos].values()))})

    def calculate_probs(self):
        self.get_row_sum()
        for i in self.all_pos:
            for j in self.all_pos[i]:
                self.all_pos[i][j] = self.all_pos[i][j]/self.row_sums[i]

    def init_prob(self, pos):
        return self.all_pos[pos]["initial"]

    def trans_prob(self, pos, prev):
        # return round(self.all_pos[prev][pos] + 0.0000000001,-3) / sum(list(self.all_pos[pos].values()))
        return self.all_pos[prev][pos]

class LineParser:
    def __init__(self, data):
        matrix.update(data[1])
        counter = 0
        while counter <= len(data[0]) - 1:
            bag.update_word(data[0][counter].strip(), data[1][counter].strip())
            counter += 1

class Solver:
    # Calculate the log of the posterior probability of a given sentence
    # with a given part-of-speech labeling. Right now just returns -999 -- fix
    # this!
    def posterior(self, model, sentence, label):
        # Posterior Prob for Simple - P(Word|POS) * P(POS)
        if model == "Simple":
            x = 0
            for i in range(len(sentence)):
                try:
                    x += math.log(bag.bag[sentence[i]].get_emission(label[i])) + math.log(bag.pos_dict[label[i]] / bag.wc)
                except:
                    x=x-10
            return x
        # Posterior Prob for HMM - P(Word|POS) * P(POS|prev POS) 
        elif model == "HMM":
            x = 0
            y = 0
            x1 = math.log(matrix.init_prob(label[0]),10)#, "initial" )
            for i in range(len(label)):
                try:
                    x += math.log(bag.bag[label[i]].get_emission(label[i]), 10)
                except:
                    x= x-10
                if i != 0:
                    y += math.log(matrix.trans_prob(label[i - 1], label[i]), 10)
            return x1 + x + y
        
        # Posterior Prob for Complex MCMC - P(Word|POS) * P(POS|prev POS) * P(POS|prev prev POS)
        elif model == "Complex":
            x = 0
            y = 0
            z = 0
            x1 = math.log(matrix.init_prob(label[0]),10)#, "initial" )
            for i in range(len(label)):
                try:
                    x += math.log(bag.bag[sentence[i]].get_emission(label[i]), 10)
                except:
                    x=x-10
                if i != 0:
                    y += math.log(matrix.trans_prob(label[i - 1], label[i]), 10)
                if i != 0 and i != 1:
                    z += math.log(matrix.trans_prob(label[i - 2], label[i]), 10)
            return x1 + x + y + z
        else:
            print("Unknown algo!")
                        
    def init_line_parser(self, data):
        # try:
        for i in data:
            LineParser(i)

    def train(self, data):
        global bag, matrix
        bag = BagOfWords()
        matrix = TheMatrix()
        self.data = data
        
        # Implementing threading to train on the dataset
        threads_arr = []
        max_threads = 4
        chunk_size = (len(data) // max_threads)
        i = 0
        while i + chunk_size <= len(data):
            t = Thread(target=self.init_line_parser,
                       args=(data[i:i + chunk_size],))
            threads_arr.append(t)
            t.start()
            i += chunk_size
        t = Thread(target=self.init_line_parser, args=(data[i:],))
        threads_arr.append(t)
        t.start()
        for i in threads_arr:
            i.join()
        matrix.calculate_probs()
        return
    
    # Implement Bayes law - Simple Chain, using emission probabilites 
    def simplified(self, sentence):
        self.simple_result = []
        for i in sentence:
            word = i.strip()
            if word in bag.bag:
                if bag.bag[word].emission_calculated:
                    # self.simple_result.append(bag.bag[word].max_pos()[0])
                    max_pos = ""
                    max_prob = 0
                    for pos in bag.pos_dict:
                        p = (bag.pos_dict[pos] / bag.wc)*(bag.bag[word].get_emission(pos))
                        if  max_prob<p:
                            max_prob = p
                            max_pos = pos
                    self.simple_result.append(max_pos)
                else:
                    for pos in list(bag.pos_dict.keys()):
                        bag.emissions(word, pos)
                    max_pos = ""
                    max_prob = 0
                    for pos in bag.pos_dict:
                        p = (bag.pos_dict[pos] / bag.wc)*(bag.bag[word].get_emission(pos))
                        if  max_prob<p:
                            max_prob = p
                            max_pos = pos
                    self.simple_result.append(max_pos)
            else:
                self.simple_result.append("noun")   # Returns noun as default for a new word
        return self.simple_result
    
    #Viterbi Algorithm for Hidden Markov Model
    def hmm_viterbi(self, sentence):
        viterbi_table = {}   
        prev_max_pos = 0
        prev_index = 0
        for i in bag.pos_dict:
            t = []
            #print(sentence[0],i,matrix.init_prob(i),bag.bag[sentence[0]].get_emission(i),matrix.init_prob(i)*bag.bag[sentence[0]].get_emission(i))
            em = bag.bag[sentence[0]].get_emission(i) if sentence[0] in bag.bag else 0.0000000001
            t.append((matrix.init_prob(i)*em,[i]))  #For the first word of all sentences
            viterbi_table.update({i:t})
            if prev_max_pos<matrix.init_prob(i)*em: #take max of initial probs*emission probs
                prev_max_pos=matrix.init_prob(i)*em
                prev_index = i
        c = 1
        for word in sentence[1:]: #For second word onwards
            prev_col = []
            for cur in bag.pos_dict:
                m = ""
                p = 0
                for prev in list(bag.pos_dict.keys()):
                        em = bag.bag[word].get_emission(cur) if word in bag.bag else 0.0000000001
                        # trans*emission*prev
                        # print(word,prev,cur,matrix.trans_prob(prev,cur),bag.bag[word].get_emission(cur),viterbi_table[prev][c-1][0],matrix.trans_prob(prev,cur)* bag.bag[word].get_emission(cur) * viterbi_table[prev][c-1][0])
                        if p<matrix.trans_prob(prev,cur)* em * prev_max_pos:  #take max of emission*transition prob*prev probabilities
                            p = matrix.trans_prob(prev,cur)* em * prev_max_pos
                t= copy.deepcopy(viterbi_table[prev_index][c-1][1])
                t.append(cur)
                # print("The best we could do for {} is {} and {}".format(sentence[:c+1],t,p))
                viterbi_table[cur].append((p, t))
                prev_col.append(p)
            c+=1
            prev_max_pos=max(prev_col)
            prev_index=list(bag.pos_dict.keys())[prev_col.index(prev_max_pos)]
        #backtracking to get the sequence
        p=0
        for cur in bag.pos_dict:
            # print(viterbi_table[cur][-1][0],viterbi_table[cur][-1][1])
            if p<viterbi_table[cur][-1][0]:
                p = viterbi_table[cur][-1][0]
                m = viterbi_table[cur][-1][1]
        return m

    def gen_sample_tags(self, sentence, sample):
        sentence_len = len(sentence)
        tags = list(bag.pos_dict.keys())
        for index in range(sentence_len):
            word = sentence[index]
            probabilities = [0] * len(matrix.pos_dict)
            for j in range(len(matrix.pos_dict)):  # try by assigning every tag
                s_2 = tags[j]  # current state is s2
                ep = bag.bag[word].get_emission(s_2) if word in bag.bag else .000000001
                if index == 0:
                    probabilities[j] = ep * matrix.init_prob(s_2)
                    continue
                elif index == 1:
                    # Transition between s2 and s1 states
                    s_1 = tags[j - 1]
                    j_k = matrix.trans_prob(s_2, s_1)  
                    probabilities[j] = j_k * ep
                    continue
                else:
                    # Trasnisition between s2 and s0 states
                    s_1 = tags[j - 1]
                    s_0 = tags[j - 2]
                    i_k = matrix.trans_prob(s_2,s_0)  
                    probabilities[j] = i_k * j_k * ep
            s = sum(probabilities)
            probabilities = [x / s for x in probabilities]
            rand = random.random()
            p_sum = 0
            for i in range(len(probabilities)):
                p = probabilities[i]
                p_sum += p
                if rand < p:
                    sample[index] = tags[i]
                    break
        return sample
    
    def complex_mcmc(self, sentence):
        sample_size = 1000
        samples = []
        sample = []
        for word in sentence:
            if word in bag.bag:
                pos, prob = bag.bag[word].max_pos()
            else:
                pos, prob = "noun", 0.0000000001
            sample.append("x")
        # burn-out samples considered
        burnout = 300
        for sample_num in range(burnout * 3):
            sample = self.gen_sample_tags(sentence, sample)
            if sample_num > burnout:
                samples.append(sample)
        count_tags = {}
        count_tags_array = []
        for j in range(len(sentence)):
            count_tags = {}
            for sample in samples:
                try:
                    count_tags[sample[j]] += 1
                except KeyError:
                    count_tags[sample[j]] = 1
            count_tags_array.append(count_tags)
        final_tags = [max(count_tags_array[i], key=count_tags_array[i].get) for i in range(len(sentence))]
        return final_tags

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")
