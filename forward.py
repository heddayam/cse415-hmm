import random
import math
import numpy as np
import scipy.stats as stats


class model:
    def __init__(self, obs=None):
        self.random_state = np.random.RandomState(0)

        self.n_states = 6
        self.mu = None
        self.covariance = None
        self.n_dims = None
        self.states = [0 for x in range(6)]
        self.observations = None

        self.pi = self.normalize(self.random_state.rand(self.n_states, 1))
        self.transitions = self.stochasticize(self.random_state.rand(self.n_states, self.n_states))
        # self.pi = [random.uniform(0, 1) for x in range(self.n_states)]
        # self.transitions = [[random.uniform(0, 1) for x in range(self.n_states)] for y in range(self.n_states)] # self.transitions
        # self.generate_emissions(obs)
        # self.train(obs)
        # self.emissions = self.emissions_probs(obs)
# def forward(obs):
#     global self.states
#
#     # initilization
#     trellis = [[0 for x in range(len(obs))] for y in range(len(self.states))]
#     for s in range(len(self.states)):
#         trellis[1][s] = self.pi[s] * self.emissions[s][0]
#     # recursion
#     for t in range(1, len(obs)):
#         for s in range(len(self.states)):
#             # print(s)
#             # print(t)
#             # print(trellis)
#             trellis[s][t] = 0
#             for sp in range(len(self.states)):
#                 # print(trellis[t-1][sp] * self.transitions[sp][s])
#                 trellis[s][t] += trellis[sp][t-1] * self.transitions[s][sp]
#                 # print(trellis)
#             # print("emission")
#             # print(self.emissions)
#             trellis[s][t] *= self.emissions[s][t]
#
#     # termination
#     pw = 0
#     for s in range(len(self.states)):
#         pw += trellis[s][len(obs)-1]
#     print(len(trellis))
#     return trellis


    def forward(self, emissions):
        # global self.pi
        # global self.emissions
        # global self.transitions

        log_likelihood = 0.
        trellis = np.zeros(emissions.shape)

        # print(self.transitions)
        for t in range(emissions.shape[1]):
            if t == 0:
                trellis[:, t] = emissions[:, t] * self.pi.ravel()
            else:
                trellis[:, t] = emissions[:, t] * np.dot(self.transitions.T, trellis[:, t - 1])

            alpha_sum = np.sum(trellis[:, t])

            if alpha_sum == 0.0:
                trellis[:, t] /= 10**-10
                log_likelihood = log_likelihood + np.log(10**-10)
            else:
                trellis[:, t] /= alpha_sum
                log_likelihood = log_likelihood + np.log(alpha_sum)
        # print(trellis.shape)
        return log_likelihood, trellis

    # def backward(obs):
    #     global self.states
    #     global self.emissions
    #     global self.transitions
    #
    #     # n = self.emissions.shape[1]
    #     n = len(obs)
    #     # b = {state: list() for state in range(len(self.states))}
    #     # for o in obs:
    #     #     for state in range(len(self.states)):
    #     #         b[state].append(0)
    #     #
    #
    #
    #     # b = np.zeros(self.emissions.shape)
    #     b = [[0 for x in range(len(obs))] for y in range(len(self.states))]
    #
    #     # construct backward trellis
    #     for state in range(len(self.states)):
    #         b[state][n-1] = 1
    #
    #     # b[:, -1] = np.ones(self.emissions.shape[0])
    #     for t in range(n-2, -1, -1):
    #         for i in range(len(self.states)):
    #             for j in range(len(self.states)):
    #                 # print(t)
    #                 # print(i)
    #                 # print(j)
    #                 # print(self.emissions.shape)
    #                 # print(self.transitions)
    #                 b[i][t] += b[j][t+1] * self.transitions[j][i] * self.emissions[j][t+1] #self.emissions[t+1][j]
    #     # print(b)
    #     return b

    def backward(self, B):
        trellis = np.zeros(B.shape)
        # print(B)
        trellis[:, -1] = np.ones(B.shape[0])
        for t in range(B.shape[1] - 1)[::-1]:
            trellis[:, t] = np.dot(self.transitions, (B[:, t + 1] * trellis[:, t + 1]))
            if np.sum(trellis[:, t]) == 0:
                trellis[:, t] /= 10**-10
            else:
                trellis[:, t] /= np.sum(trellis[:, t])
        # print(trellis)
        return trellis

    # def posterior(obs):
    #     global self.states
    #     global self.emissions
    #     global self.transitions
    #
    #     n = len(obs)
    #     # initialize forward, backward, and posterior trellises
    #     f = forward(obs)
    #     b = backward(obs)
    #     p = {state: list() for state in range(len(self.states))}
    #     for o in obs:
    #         for state in range(len(self.states)):
    #             p[state].append(0)
    #
    #     # total probability of sequence O
    #     # print(f)
    #     O_prob = math.fsum([f[state][n-1] for state in range(len(self.states))])
    #
    #     # build posterior trellis
    #     for state in range(len(self.states)):
    #         for t in range(n):
    #             p[state][t] = (f[state][t] * b[state][t]) / O_prob
    #
    #     return p

    # def viterbi(obs, state):
    #     trellis_viterbi = [[0 for x in range(len(obs))] for y in range(len(self.states))]
    #     trellis_back = [[0 for x in range(len(obs))] for y in range(len(self.states))]
    #
    #     for s in range(len(state)):
    #         trellis_viterbi[s][1] = self.pi[s] * self.emissions[s][obs[1]]
    #
    #     for t in range(2, len(obs)):
    #         for s in range(len(state)):
    #             trellis_viterbi[s][t] = 0
    #             for sp in range(len(state)):
    #                 temp = trellis_viterbi[sp][t-1] * self.transitions[sp][s]
    #                 if temp > trellis_viterbi[s][t-1]:
    #                     trellis_viterbi[s][t - 1] = temp
    #                     trellis_back[s][t - 1] = sp
    #
    #             trellis_viterbi[s][t] *= self.emissions[s][obs[t]]
    #
    #     max_s = None
    #     viterbi_max = 0
    #     for s in range(len(state)):
    #         if trellis_viterbi[s][len(obs)-1] > viterbi_max:
    #             max_s = s
    #             viterbi_max = trellis_viterbi[s][len(obs)-1]
    #
    #     i = len(obs)-1
    #     backtrace = [0 for x in range(len(obs)+1)]
    #     while i > 0:
    #         backtrace[i] = max_s
    #         max_s = trellis_back[i][max_s]
    #         i -= 1
    #
    #     return backtrace


    def emissions_probs(self, obs):
        # global self.covariance
        # global self.mu
        # global self.n_states
        # global self.emissions

        obs = np.atleast_2d(obs)
        B = np.zeros((self.n_states, obs.shape[1]))
        # print(self.covariance)
        for s in range(self.n_states):  # Probability of getting the observation (o1,o2,...oT) when it is in state "s"
            # Needs scipy 0.14
            np.random.seed(self.random_state.randint(1))
            # print(stats.multivariate_normal.pdf(obs.T, mean=self.mu[:, s].T, cov=self.covariance[:, :, s].T))
            B[s, :] = stats.multivariate_normal.pdf(obs.T, mean=self.mu[:, s].T, cov=self.covariance[:, :, s].T)
        # self.emissions = B
        # print(B)
        return B

    def generate_emissions(self, obs):
        # global self.n_dims
        # global self.mu
        # global self.covariance
        # global self.n_states
        # global self.pi
        # global self.transitions

        if self.n_dims is None:
            self.n_dims = obs.shape[0]
        if self.mu is None:
            subset = np.random.choice(np.arange(self.n_dims), size=self.n_states, replace=False)
            self.mu = obs[:, subset]
        if self.covariance is None:
            self.covariance = np.zeros((self.n_dims, self.n_dims, self.n_states))
            self.covariance += np.diag(np.diag(np.cov(obs)))[:, :, None]

        # print(self.n_dims)
        # print(self.mu)
        # print(self.covariance)


    def normalize(self, x):
        if np.sum(x) == 0:
            return (x + (x == 0)) / 10**-10
        else:
            return (x + (x == 0)) / np.sum(x)



    def stochasticize(self, x):
        return (x + (x == 0)) / np.sum(x, axis=1)

    def forward_backward(self, obs):
        # global self.states
        # global self.transitions
        # global self.emissions
        # global self.pi
        # global self.mu
        # global self.covariance

        obs = np.atleast_2d(obs)
        emissions = self.emissions_probs(obs)
        # T = obs.shape[1]
        # print(emissions)
        log_likelihood, trellis_fwd = self.forward(emissions)
        trellis_bwd = self.backward(emissions)

        xi_sum = np.zeros((self.n_states, self.n_states))
        gamma = np.zeros((self.n_states, obs.shape[1]))

        for t in range(obs.shape[1] - 1):
            partial_sum = np.asarray(self.transitions) * np.dot(trellis_fwd[:,t], (trellis_bwd[:,t] * emissions[:,t + 1]).T)
            xi_sum += self.normalize(partial_sum)
            partial_g = trellis_fwd[:, t] * trellis_bwd[:, t]
            gamma[:, t] = self.normalize(partial_g)

        partial_g = trellis_fwd[:, -1] * trellis_bwd[:, -1]
        gamma[:, -1] = self.normalize(partial_g)

        new_pi = gamma[:, 0]
        new_transitions = self.stochasticize(xi_sum)

        new_mu = np.zeros((self.n_dims, self.n_states))
        new_covs = np.zeros((self.n_dims, self.n_dims, self.n_states))

        gamma_state_sum = np.sum(gamma, axis=1)
        # Set zeros to 1 before dividing
        gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)
        # print(gamma)
        for s in range(self.n_states):
            gamma_obs = obs * gamma[s, :]
            new_mu[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]
            # print(np.sum(gamma_obs, axis=1))
            # print(gamma_state_sum[s])
            partial_covs = np.dot(gamma_obs, obs.T) / gamma_state_sum[s] - np.dot(new_mu[:, s], new_mu[:, s].T)
            # Symmetrize
            partial_covs = np.triu(partial_covs) + np.triu(partial_covs).T - np.diag(partial_covs)

        # Ensure positive semidefinite by adding diagonal loading
        new_covs += .01 * np.eye(self.n_dims)[:, :, None]

        self.pi = new_pi
        self.mu = new_mu
        self.covariance = new_covs
        self.transitions = new_transitions
        # print(self.transitions)
        # return log_likelihood

    def transform(self, obs):
        # global self.emissions

        if len(obs.shape) == 2:
            B = self.emissions_probs(obs)
            # self.emissions = self.emissions_probs(obs)
            log_likelihood, _ = self.forward(B)
            return log_likelihood
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            out = np.zeros((count,))
            for n in range(count):
                B = self.emissions_probs(obs[n, :, :])
                log_likelihood, _ = self.forward(B)
                out[n] = log_likelihood
            return out

    def train(self, obs, n_iter=15):
        # Support for 2D and 3D arrays
        # 2D should be n_features, n_dims
        # 3D should be n_examples, n_features, n_dims
        # For example, with 6 features per speech segment, 105 different words
        # this array should be size
        # (105, 6, X) where X is the number of frames with features extracted
        # For a single example file, the array should be size (6, X)
        if len(obs.shape) == 2:
            for i in range(n_iter):
                self.generate_emissions(obs)
                log_likelihood = self.forward_backward(obs)
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            for n in range(count):
                for i in range(n_iter):
                    self.generate_emissions(obs[n, :, :])
                    log_likelihood = self.forward_backward(obs[n, :, :])
        return self


    # n = len(obs)
    # f = forward(obs)
    # b = backward(obs)
    # p = posterior(obs)
    # 
    # emission_p = {}
    # for emission in range(len(self.emissions)): emission_p[emission] = {}
    # transition_p = {}
    # for state in range(len(self.states)): transition_p[state] = {}
    # self.pi_p = {}
    # 
    # # construct emission_p
    # for state in range(len(self.states)):
    #     den = math.fsum([p[state][t] for t in range(n)])
    #     for emission in self.emissions:
    #         v = 0
    #         for t in range(n):
    #             if obs[t].all() == emission.all():
    #                 v += p[state][t]
    #         emission_p[emission][state] = v / den
    # 
    # # construct transition_p
    # p_O = math.fsum([f[s][n-1] for s in range(len(self.states))])
    # for given in range(len(self.states)):
    #     den = math.fsum([p[given][t] for t in range(n)])
    #     for to in range(len(self.states)):
    #         v = 0
    #         for t in range(1, n):
    #             v += (  f[given][t-1] *
    #                     b[to][t] *
    #                     self.transitions[to][given] *
    #                     self.emissions[obs[t]][to]
    #                     ) / p_O
    #         transition_p[to][given] = v / den
    # 
    # # construct self.pi_p
    # for state in range(len(self.states)):
    #     self.pi_p[state] = p[state][0]
    # 
    # # new_hmm = HMM(self.states, hmm._self.emissions)
    # self.emissions = emission_p
    # self.transitions = transition_p
    # self.pi = self.pi_p


