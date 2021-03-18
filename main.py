# main.py
# CSE 415
# Project Milestone c
# June 04, 2019
# Mourad Heddaya

# Hidden Markov Model Part of Speech Tagging

import HMM as t
import numpy as np

if __name__ == '__main__':
    m = t.hmmModel()
    # obs = "The President has no idea what they are doing !".split()
    while True:
        print("Sentence to parse:")
        s = input("> ")
        obs = s.strip().split()
        prediction, bestpath_prob = m.viterbi(obs)
        for i in range(len(obs)):
            print(obs[i] + "/" + prediction[i] + " ", end="", flush=True)
        print()
        print("Log Likelihood = {:+.2f}".format(np.log(bestpath_prob)))
        print()


        # print(prediction)

    # obs = "Because I had to catch the train , and as we were short on time , I forgot to pack my brush for our vacation .".split()
    # She returned the computer after she noticed it was damaged .

    # prediction, bestpath_prob = m.viterbi(obs)
    # print(prediction)
    # m.forward(obs)
    # print(m.forward(obs)[1])
    # m.baumWelch(obs)
    # m.viterbi(obs)
    # print(prediction)
    # print(m.forward(obs)[1])


