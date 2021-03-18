# estimate_params.py
# CSE 415
# Project Milestone C
# June 04, 2019
# Mourad Heddaya

# Hidden Markov Model Part of Speech Tagging

import numpy as np
import glob

def modify(lines, output):
    for l in lines:
        output.write("[")
        for token in l.strip().split():
            parsed = token.strip().rsplit("_", 1)
            if len(parsed) == 2:
                if parsed[0] == "\\": parsed[0] = "\\\\"
                parsed[1] = parsed[1].replace('"', "'")
                parsed[0] = parsed[0].replace('"', "'")
                if "//" in parsed[0]:
                    parsed[0] = "URL"
                output.write('["' + parsed[0] + '", "' + parsed[1] + '"], ')
        output.write("]\n")

def read_raw():
    output = open("tagged_data.json", "w")

    path = 'tagged/*.txt'
    files = glob.glob(path)
    for file in files:
        f = open(file, 'r')
        modify(f.readlines(), output)
        f.close()

    output.close()

file = open(r"tagged_data.json", 'r', errors='ignore')
lines = file.readlines()

sentences = []
tags = []
words = []
word_tag_dict = dict()
tag_tag_dict = dict()
for l in lines:
    parsed = eval(l)
    prev = ("", 'BEGIN')
    if 'BEGIN' not in tags:
        tags.append('BEGIN')
    for pair in parsed:
        if pair[0] not in words:
            words.append(pair[0])
        if pair[1] not in tags:
            tags.append(pair[1])

        if (pair[0], pair[1]) not in word_tag_dict:
            word_tag_dict[(pair[0], pair[1])] = 1
        else:
            word_tag_dict[(pair[0], pair[1])] += 1

        if (prev[1], pair[1]) not in tag_tag_dict:
            tag_tag_dict[(prev[1], pair[1])] = 1
        else:
            tag_tag_dict[(prev[1], pair[1])] += 1
        prev = pair

    # print(parsed[len(parsed) - 1])
    # if parsed[len(parsed) - 1][1] != ',':
    #     if ',' not in tags:
    #         tags.append(',')
    #     if '.' not in words:
    #         words.append('.')
    #     if ('.', ',') not in word_tag_dict:
    #         word_tag_dict[('.', ',')] = 1
    #     else:
    #         word_tag_dict[('.', ',')] += 1
    #
    #     if (parsed[len(parsed) - 1][1], ',') not in tag_tag_dict:
    #         tag_tag_dict[(parsed[len(parsed) - 1][1], ',')] = 1
    #     else:
    #         tag_tag_dict[(parsed[len(parsed) - 1][1], ',')] += 1
file.close()

unique_b = {}
for tag in tags:
    unique_b[tag] = []
    for key, value in word_tag_dict.items():
        if key[1] == tag:
            unique_b[tag].append((key[0], value))

unique_t2t = {}
for tag in tags:
    unique_t2t[tag] = []
    for key, value in tag_tag_dict.items():
        if key[0] == tag:
            unique_t2t[tag].append((key[1], value))

def estimate_emissions():
    b = np.zeros((len(words), len(tags)))

    for key, value in unique_b.items():
        total = sum(i for _, i in value)
        for p in value:
            b[words.index(p[0]), tags.index(key)] = float(p[1]) / total
    return b

def estimate_transitions():
    a = np.zeros((len(tags), len(tags)))
    for key, value in unique_t2t.items():
        total = sum(i for _, i in value)
        for p in value:
            a[tags.index(key), tags.index(p[0])] = float(p[1] / total)
    return a

def estimate_pi():
    pi = np.zeros(len(tags))
    beginning = unique_t2t['BEGIN']
    total = sum(i for _, i in beginning)
    for p in beginning:
        pi[tags.index(p[0])] = float(p[1] / total)

    return pi

def get_tags():
    global tags
    return tags

def get_words():
    global words
    return words
