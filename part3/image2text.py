#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2020)
#
import copy
import datetime
import heapq
import json
import math
from threading import Lock, Thread

from PIL import Image, ImageDraw, ImageFont
import sys

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

# for simple, things we can do :

#   find matched 'x' and give weights for matches

#   use techniques similar to Jaro Similarity > some margin say, 0.8 for the whole
#         similarity of entire char being some train char

#   find grid density and look for similar grid desnisty
#
# 2*2, 3*3 */spaces matching grids/total grids

#   find find % row/column wise matches

#   'Thinning' the image to smaller ratios and calulate the above,
#       then use extrapolation techniques to figureout the character
#   count the transformations needed

# # observed -> is array of arrays for any given char
#
# # hidden   -> chars
#
# # p(o|h) = p(h|o)*p(o)/p(h)
#
# # p(h|o) =	compare()[h][0]/total

# p(o)	 = 	1

# p(h)	 = 	given char/training

# for each word
# transition -> given current char, what is the prob for next char (as inferred from the bc.train after removing the 12 char__ from the text)
# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
#  transition*emission

# It is so ordered
# 1t 1s s0
# input is bc.train
# find which char is most probably next
#   transition -> from bc.train
#   emission? probablity observed| hidden
# observed
#

class TheMatrix:

    def __init__(self, char___dict):
        self.all_char__ = {}
        self.char___dict_cols = {"initial": pow(10, -10)}
        for i in char___dict:
            self.char___dict_cols.update({i: pow(10, -10)})
        for i in char___dict:
            self.all_char__.update({i: copy.deepcopy(self.char___dict_cols)})
        self.lock = Lock()
        self.wc = 0
        self.cc = 0

    # decorator
    def _lock(func):
        def wrapper(self, *args, **kwargs):
            self.lock.acquire()
            r = func(self, *args, **kwargs)
            self.lock.release()
            return r

        return wrapper

    @_lock
    def update(self, char___list):
        self.wc += 1
        self.cc += 1
        try:
            # Super
            # S
            t = self.all_char__[char___list[0]]["initial"]
            self.all_char__[char___list[0]]["initial"] = t + 1
        except Exception as e:
            # this too shall pass
            pass

        for i in range(len(char___list) - 1):
            self.cc += 1
            try:
                c = self.all_char__[char___list[i]][char___list[i + 1]]
                temp = self.all_char__[char___list[i]]
                temp.update({char___list[i + 1]: c + 1})
                self.all_char__.update({char___list[i]: temp})
            except Exception as e:
                # this too shall pass
                pass

    def init_prob(self, char):
        # probability of this char starting at 0th position/total number of times this char occurs
        return (self.all_char__[char]["initial"] / sum(list(self.all_char__[char].values())))

    def trans_prob(self, char, prev):
        # number of times the this char has occured given the previous char/ total occurances of this char
        return (self.all_char__[prev][char] / sum(list(self.all_char__[char].values())))


class Character:
    def __init__(self, character, list_of_strings):
        self.character = character
        self.list_of_strings = list_of_strings
        self.match_weight = 0.85
        self.mismatch_space_weight = 0.01
        self.mismatch_asterisk_weight = 0.04
        self.space_weight = 0.1

    def __len__(self):
        return len(self.list_of_strings)

    def __getitem__(self, row):
        return self.list_of_strings[row]

    def compare(self, other, noise):
        space_match = 0
        asterick_match = 0
        space_mismatch = 0
        asterick_mismatch = 0

        for i in range(0, len(other)):
            for j in range(0, len(other[i])):
                if self.list_of_strings[i][j] == other[i][j]:
                    if other[i][j] == " ":
                        # space in both arrays
                        space_match += 1
                    else:
                        # same '*' in both arrays
                        asterick_match += 1
                else:
                    # if there is a " " and "*" or vice-versa, it is a mismatch
                    if other[i][j] == " ":
                        # if the train image has a * but test has a space.
                        # this will be fine
                        asterick_mismatch += 1
                    else:
                        # if the train image has a space but test has a *.
                        # this must have higher weight
                        space_mismatch += 1
        # p(h|o) = p(o|h)
        #        = P(Training letter| Test letter)
        #        =
        return {"asterick_match": asterick_match, "space_mismatch": space_mismatch, "asterick_mismatch": asterick_mismatch,
                "spaces": space_match,"prob": math.pow(0.9, 0.9*asterick_match) * math.pow(0.1, space_mismatch) * math.pow(0.6,space_match) * math.pow(0.4, 0.9*asterick_mismatch)
               }



def read_data(fname):
    exemplars = []
    file = open(fname, 'r');
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += [(data[0::2])]
    return exemplars


def get_emission(chars):
    emissions = {}
    for j in train_chars:
        emissions.update({j: train_chars[j].compare(chars, 0.1)["prob"]})
    return emissions

def transtions(previous_state,current_letter):
    heaps = []
    for index_of_previous_char, previous_letter in enumerate(train_letters):
        trans_prev_current = -math.log(
            trans[list(train_chars.keys()).index(previous_letter)][list(train_chars.keys()).index(current_letter) + 1]) \
                             + previous_state[ord(previous_letter)][0]
        heaps.append([trans_prev_current, previous_state[ord(previous_letter)][1] + [current_letter]])
    return  heaps,previous_state

def hmm_viterbi(init,ems):
    max_asci_code = 255
    current_state,previous_state = [0]*max_asci_code, [0]*max_asci_code
    for index_of_state, state in enumerate(test_letters):
        for index, current_letter in enumerate(train_letters):
            if index_of_state == 0:
                current_state[ord(current_letter)] = [-math.log(init[index]) - math.log(ems[0][index]), [current_letter]]
            else:
                heaps, previous_state = transtions(previous_state,current_letter)
                heapq.heapify(heaps)
                max_from_transition = heapq.heappop(heaps)
                result = max_from_transition[0] - math.log(ems[index_of_state][index])
                current_state[ord(current_letter)] = [result, max_from_transition[1]]
        previous_state = copy.deepcopy(current_state)
        current_state = [None] * max_asci_code
    f = [i for i in previous_state if i is not  None]
    heapq.heapify(f)
    return heapq.heappop(f)[1]


#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
train_chars = {}

for i in train_letters:
    train_chars.update({i: Character(i, train_letters[i])})
matrix = TheMatrix(list(train_chars.keys()))

for i in open(train_txt_fname).readlines():
    matrix.update(i)

trans = []
for i in list(matrix.all_char__.keys()):
    trans.append(list(matrix.all_char__[i].values()))
init = []
for i in train_chars:
    init.append(matrix.init_prob(i))
test_chars = []
ems = []
ems = []

for i in test_letters:
    p = []
    for j in train_chars:
        p.append(train_chars[j].compare(i, 0)["prob"])
    test_chars.append(list(train_chars.keys())[p.index(max(p))])
    ems.append(list(get_emission(i).values()))

print("Simple: "+"".join(test_chars))
print("   HMM: " + "".join(hmm_viterbi(init, ems)))
