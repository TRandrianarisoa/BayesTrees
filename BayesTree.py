"""Base classes for Bayesian Tree-based methods"""

import numpy as np
from random import randint, uniform


class Node:
    def __init__(self, m, a, dimension, points):
        self.p = None  # split variable if internal node
        self.v = None  # value for split variable if internal node

        self.m = m  # mean for leaf distribution (likelihood)
        self.a = a  # scale of mean distribution (likelihood)

        self.children = None  # children nodes if internal node
        self.dim = dimension  # data space dimension
        self.identifier = ''  # identifier of the node in the tree
        self.points = points  # points falling in the cell node

    def add_children(self, pred, val):
        ''' Add a split on this node '''
        self.p = pred
        self.v = val
        left = Node(self.m, self.a,
                    self.dim, np.array([pt for pt in self.points if pt[0][pred] < val]))
        left.identifier = self.identifier+'0'
        right = Node(self.m, self.a,
                     self.dim, np.array([pt for pt in self.points if pt[0][pred] >= val]))
        right.identifier = self.identifier + '1'
        self.children = np.array([left, right])

    def delete_children(self):
        ''' Transform in leaf '''
        self.p = None
        self.v = None
        self.children = None

    def is_terminal(self):
        return self.children is None

    def change_split_var(self):
        self.p = randint(0, self.dim-1)
        self.v = uniform(0, 1)
        self.modif_part()

    def change_split_rule(self):
        self.v = uniform(0, 1)
        self.modif_part()

    def modif_part(self):
        if self.children is None:
            pass
        else:
            left, right = self.children[0], self.children[1]
            left.points = np.array([pt for pt in self.points if pt[0][self.p] < self.v])
            left.modif_part()
            right.points = np.array([pt for pt in self.points if pt[0][self.p] >= self.v])
            right.modif_part()
            self.children = np.array([left, right])

    def prior(self, n, alpha, beta):
        if self.children is None:
            return 1 - alpha * (1+n)**(-beta)
        else:
            return self.children[0].prior(n+1, alpha, beta) * self.children[0].prior(n+1, alpha, beta)

    def likelihood(self):
        if len(self.points)==0:
            return None
        if self.is_terminal():
            return np.array([self.likelihood_parameters()])
        else:
            left, right = self.children[0].likelihood(), self.children[1].likelihood()
            if left is None:
                return right
            elif right is None:
                return left
            else:
                return np.append(self.children[0].likelihood(), self.children[1].likelihood(), axis=0)

    def likelihood_parameters(self):
        n_i = len(self.points)
        if n_i == 0:
            return []
        return [
            n_i,
            (n_i-1)*np.var(self.points[:, 1]),
            n_i*self.a*(np.mean(self.points[:, 1])-self.m)**2/(n_i+self.a)
        ]


class Tree:
    def __init__(self, m, a, nu, lambd, points, alpha, beta):
        dimension = len(points[0])

        self.root = Node(m, a, dimension, points)  # root node
        self.dim = dimension   # data space dimension
        self.internal_nodes = []  # identifiers of internal nodes
        self.leaf = ['']  # identifiers of leaves

        # prior parameters
        self.alpha = alpha
        self.beta = beta

        # parameters for inverse gamma function
        self.nu = nu
        self.lambd = lambd

    def get_node(self, node, node_identifier):
        if node_identifier == '':
            return node
        elif node_identifier == '0':
            return node.children[0]
        elif node_identifier == '1':
            return node.children[1]
        else:
            char, node_identifier = node_identifier[0], node_identifier[1:]
            if char == '0':
                return self.get_node(node.children[0], node_identifier)
            else:
                return self.get_node(node.children[1], node_identifier)

    def add_leaves(self, pred, threshold, node_identifier):
        assert node_identifier in self.leaf, 'Leaves are appended to previous leaves'

        node_ = self.get_node(self.root, node_identifier)
        node_.add_children(pred, threshold)

        self.internal_nodes.append(node_identifier)
        self.leaf.remove(node_identifier)
        self.leaf.extend([node_identifier+'0',  node_identifier+'1'])

    def grow(self):
        node_identifier = np.random.choice(self.leaf)
        p = randint(0, self.dim-1)
        v = uniform(0, 1)

        self.add_leaves(p, v, node_identifier)

    def prune(self):
        assert len(self.internal_nodes) > 0, 'Cannot prune if the Tree is only a root'
        node_identifier = np.random.choice(self.leaf)[:-1]

        self.leaf = [leaf for leaf in self.leaf if not leaf.startswith(node_identifier)]
        self.leaf.append(node_identifier)

        self.internal_nodes = [nd for nd in self.internal_nodes if not nd.startswith(node_identifier)]

        node_ = self.get_node(self.root, node_identifier)
        node_.delete_children()

    def change_split_var(self):
        node_identifier = np.random.choice(self.internal_nodes)
        node_ = self.get_node(self.root, node_identifier)
        node_.change_split_var()

    def change_split_rule(self):
        node_identifier = np.random.choice(self.internal_nodes)
        node_ = self.get_node(self.root, node_identifier)
        node_.change_split_rule()

    def likelihood(self):
        list_param = self.root.likelihood()
        prod = np.sqrt(np.prod(list_param[:, 0]+self.root.a))
        sum_ = np.sum(list_param[:, 1] + list_param[:, 2])
        likelihood = (sum_+self.nu*self.lambd)**((len(self.root.points)+self.nu)/2)*self.root.a**(len(list_param)/2)/prod

        return likelihood[0]

    def prior(self):
        return self.root.prior(0, self.alpha, self.beta) * (1/self.dim)**len(self.internal_nodes)

    def count_points_nodes(self):
        for node in self.internal_nodes:
            print('Internal node %s has %.0f points' % (node, len(self.get_node(self.root, node).points)))
        for node in self.leaf:
            print('Leaf %s has %f points' % (node, len(self.get_node(self.root, node).points)))
