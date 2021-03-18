# HMM.py
# CSE 415
# Project Milestone C
# June 04, 2019
# Mourad Heddaya

# Hidden Markov Model Part of Speech Tagging

import numpy as np
import estimate_params as estimator
from graphviz import Digraph

class hmmModel:
    def __init__(self, a=None, b=None, pi=None, states=None, words=None):
        if a is None:
            estimator.read_raw()
            self.train()
            self.states = estimator.get_tags()
            self.words = estimator.get_words()
        else:
            self.a = a
            self.b = b
            self.pi = pi
            self.states = states
            self.words = words

    def train(self):
        self.a = estimator.estimate_transitions()
        self.b = estimator.estimate_emissions()
        self.pi = estimator.estimate_pi()

    def viterbi(self, obs):
        N = len(self.states)
        T = len(obs)
        viterbi = np.zeros((N, T))

        backpointer = np.zeros((N, T))
        for s in range(N):
            if obs[0] in self.words:
                viterbi[s, 0] = self.pi[s] * self.b[self.words.index(obs[0]), s]
            else:
                viterbi[s, 0] = self.pi[s] * self.b[self.words.index("/"), s]
            backpointer[s, 0] = 0

        for t in range(1, T):
            for s in range(N):
                max_val = float('-inf')
                max_arg = 0
                for sp in range(N):
                    if obs[t] in self.words:
                        tmp = viterbi[sp, t-1] * self.a[sp, s] * self.b[self.words.index(obs[t]), s]
                    else:
                        tmp = viterbi[sp, t - 1] * self.a[sp, s] * self.b[self.words.index("/"), s]
                    if tmp > max_val:
                        max_val = tmp
                        max_arg = sp
                viterbi[s, t] = max_val
                backpointer[s, t] = max_arg

        best_path_prob = 0
        best_path_ptr = 0
        for s in range(N):
            tmp = viterbi[s, T-1]
            if tmp > best_path_prob:
                best_path_prob = tmp
                best_path_ptr = s

        bestpath = np.zeros(T)
        for t in range(T-1, -1, -1):
            bestpath[t] = best_path_ptr
            best_path_ptr = backpointer[int(best_path_ptr), t]

        bestpath = bestpath.tolist()
        bestpath = [int(i) for i in bestpath]
        prediction = np.asarray(self.states)[bestpath]
        self.graph(obs, viterbi, backpointer, prediction, bestpath)
        return prediction, best_path_prob

    def forward(self, obs):
        N = len(self.states)
        T = len(obs)
        forward = np.zeros((N, T))

        for s in range(N):
            forward[s,0] = self.pi[s] * self.b[self.words.index(obs[0]), s]

        for t in range(1, T):
            for s in range(N):
                for sp in range(N):
                    forward[s, t] += forward[sp, t-1] * self.a[sp, s]
                forward[s, t] *= self.b[self.words.index(obs[t]), s]

        forward_prob = 0
        for s in range(N):
            forward_prob += forward[s, T-1]

        # self.graph(obs, forward, forward)

        return forward, forward_prob

    def backward(self, obs):
        N = len(self.states)
        T = len(obs)
        backward = np.zeros((N, T))

        for s in range(N):
            backward[s, T-1] = 1

        for t in range(T-2, -1, -1):
            for s in range(N):
                for sp in range(N):
                    backward[s, t] += backward[sp, t+1] * self.a[s, sp]
                backward[s, t] *= self.b[self.words.index(obs[t+1]), s]

        backward_prob = 0
        for s in range(N):
            backward_prob += backward[s, 0] * self.pi[s] * self.b[self.words.index(obs[0]), s]
        return backward, backward_prob

    def gamma(self, i, t, alpha, beta):
        N = len(self.states)
        # T = len(obs)
        # N,T
        # print(alpha.shape)
        # print(alpha[i, t])
        n = alpha[i, t] * float(beta[i, t])
        d = 0
        for j in range(N):
            d += float(alpha[j, t]) * beta[j, t]
        if d == 0:
            return float(n)/10**-10
        return float(n)/d

    def xi(self, i, j, t, alpha, beta, obs):
        N = len(self.states)
        n = float(alpha[i, t]) * self.a[i, j] * self.b[self.words.index(obs[t+1]), j] * beta[j, t+1]
        d = 0
        for j in range(N):
            d += float(alpha[j, t]) * beta[j, t]

        if d == 0:
            return float(n) / 10 ** -10
        return float(n) / d

    def baumWelch(self, obs):
        T = len(obs)
        N = len(self.states)
        M = len(self.words)
        alpha = np.zeros((N, T))
        beta = np.zeros((N, T))
        newPi = np.zeros(N)
        newA = np.zeros((N, N))
        newB = np.zeros((M, N))
        done = False
        count = 0
        while not done:
            count+=1
            alpha = self.forward(obs)[0]
            beta = self.backward(obs)[0]
            for i in range(N):
                newPi[i] = self.gamma(i, 1, alpha, beta)

            for i in range(N):
                for j in range(N):
                    sum_xi = 0
                    sum_gamma = 0
                    for t in range(T-1):
                        sum_xi += self.xi(i, j, t, alpha, beta, obs)
                        sum_gamma += self.gamma(i, t, alpha, beta)

                    if sum_gamma == 0:
                        newA[i, j] = float(sum_xi)/10**-10
                    else: newA[i, j] = float(sum_xi)/sum_gamma

            for j in range(N):
                for k in range(M):
                    n = 0
                    d = 0
                    for t in range(T):
                        interD = self.gamma(j, t, alpha, beta)
                        if self.words[k] == obs[t]:
                            n += interD
                        d += interD

                    if d == 0:
                        newB[k][j] = float(n) / 10**-10
                    else: newB[k][j] = float(n) / d

            newHmm = hmmModel(newA, newB, newPi, self.states, self.words)

            oldProb = 0
            newProb = 0
            for i in range(N):
                oldProb += alpha[i][T-1]
            newProb = newHmm.forward(obs)[1]

            if newProb > oldProb:
                self.a = newA
                self.b = newB
                self.pi = newPi
                done = True
            else:
                done = True



    def graph(self, obs, viterbi, matrix, prediction, bestpath=None):
        dot = Digraph(comment='Viterbi')

        dot.attr(rankdir='LR')
        dot.attr('node', shape='circle')

        for t in range(matrix.shape[1]-1, -1, -1):
            for r in range(matrix.shape[0]):
                if matrix[r, t] != 0.0:
                    prev = matrix[r, t]
                    if bestpath is not None and r == bestpath[t]:
                        dot.node(self.states[int(r)] + " (" + obs[t] + " " + str(t) + ")",
                                 color="purple", fillcolor='#E6E6FA', style='filled')
                        dot.node(self.states[int(prev)] + " (" + obs[t - 1] + " " + str(t-1) + ")",
                                 color="purple", fillcolor='#E6E6FA', style='filled')
                        dot.edge(self.states[int(prev)] + " (" + obs[t - 1] + " " + str(t-1) + ")",
                                 self.states[int(r)] + " (" + obs[t] + " " + str(t) + ")",
                                 label="{:+.2f}".format(np.log(viterbi[r, t])),
                                 color='purple', penwidth="2.0")
                    else:
                        dot.node(self.states[int(r)] + " (" + obs[t] + " " + str(t) + ")",
                                 style='dashed')
                        dot.node(self.states[int(prev)] + " (" + obs[t - 1] + " " + str(t-1) + ")",
                                 style='dashed')
                        dot.edge(self.states[int(prev)] + " ("+obs[t-1] + " " + str(t-1) + ")",
                                 self.states[int(r)] + " ("+obs[t] + " " + str(t) + ")",
                                 label="{:+.2f}".format(np.log(viterbi[r,t])),
                                 style="dashed")

        dot.node(prediction[0] + " (" + obs[0] + " " + str(0) + ")",
                 color="purple", fillcolor='#E6E6FA', style='filled')

        if bestpath is None:
            dot.render('test-output/forward.gv', view=True)
        else:
            dot.render('test-output/viterbi.gv', view=True)
