""" Reversible jumps with Bayesian Trees """

from BayesTree import Tree
from numpy.random import choice, uniform
from copy import deepcopy
from tqdm import tqdm, tqdm_notebook


def RJBT(points, rounds, m, a, nu, lambd, alpha, beta, print_n=400):

    T = Tree(m, a, nu, lambd, points, alpha, beta)
    Trees = [T]

    likelihood = T.likelihood()
    prior = T.prior()

    for i in tqdm_notebook(range(rounds)):

        if i % print_n == 0:
            T.count_points_nodes()

        moves = ['GROW', 'PRUNE', 'VAR', 'RULE']
        if len(T.leaf) == 1:
            prob = [1, 0, 0, 0]
        else:
            prob = [0.25, 0.25, 0.25, 0.25]

        move = choice(moves, 1,
                      p=prob)

        T_ = deepcopy(T)

        if move == 'VAR':
            T.change_split_var()
            likelihood_new = T.likelihood()

            likelihood_r = likelihood_new/likelihood
            prior_r = T.prior()/prior

            alpha = min(1, likelihood_r*prior_r)

        elif move == 'RULE':
            T.change_split_rule()
            likelihood_new = T.likelihood()

            likelihood_r = likelihood_new/likelihood
            prior_r = T.prior()/prior

            alpha = min(1, likelihood_r*prior_r)

        elif move == 'PRUNE':

            k = len(T.leaf)

            distrib_ratio = 0.25 * (k == 2) + 1 * (k != 2)
            distrib_ratio *= (k-1)*T.dim / k

            T.prune()
            likelihood_new = T.likelihood()

            likelihood_r = likelihood_new/likelihood
            prior_r = T.prior()/prior

            alpha = min(1, likelihood_r*prior_r*distrib_ratio)

        elif move == 'GROW':

            k = len(T.leaf)

            distrib_ratio = 4 * (k == 1) + 1 * (k != 1)
            distrib_ratio *= (k + 1) / (k * T.dim)

            T.grow()
            likelihood_new = T.likelihood()

            likelihood_r = likelihood_new / likelihood
            prior_r = T.prior() / prior

            alpha = min(1, likelihood_r * prior_r * distrib_ratio)

        if uniform() < alpha:
            Trees.append(deepcopy(T))

            prior = T.prior()
            likelihood = T.likelihood()

        else:
            T = T_

            Trees.append(deepcopy(T))

    return Trees
