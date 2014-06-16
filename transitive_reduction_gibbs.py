from copy import deepcopy

from numpy import *


def toggle(theta, i):
    #toggle ith element of theta
    new_theta = deepcopy(theta)
    new_theta[i] = float(not(new_theta[i]))
    return new_theta


def vector2adj(theta, g):
    sel = logical_and(g>0, tril(ones(g.shape), 0))
    adj = zeros(g.shape)
    adj[sel] = theta
    adj[transpose(sel)] = theta
    return adj


def adj2vector(g):
    sel = logical_and(g>0, tril(ones(g.shape), 0))
    return g[sel]
                      
    
def differential(func, theta, min_slope=1.0e-8):
    #calculates product of differentials of func over theta in all dimensions
    value = func(theta)
    dif = 1.0
    for i in range(0, len(theta)):
        new_theta = toggle(theta, i)
        d_theta = new_theta[i] - theta[i]
        d_value = func(new_theta)-value
        slope = d_value/d_theta
        print slope
        if not slope:
            slope = min_slope
        elif slope < min_slope:
            min_slope = slope
        dif = dif*slope
        assert dif
    return abs(dif)

        
class gibbs_sampler(object):

    def __init__(self, theta_0, dist, ignore = 100, total = 100):
        self.theta = theta_0   #initial theta
        self.dist = dist   #desired distribution function, dist(theta) is proportional to the desired possibility of picking theta
        self.prob = self.dist(self.theta)   #possibility of sampling theta, inverse of its weight
        self.total = total+ignore
        self.count = 0
        for i in range(0, ignore):
            self.next()

    def next(self):
        if self.count >= self.total:
            raise StopIteration
        for j in range(0, 50):
            for i in range(0, len(self.theta)):
                new_theta = toggle(self.theta, i)
                new_prob = self.dist(new_theta)
                cond_p = new_prob/(new_prob+self.prob)
                if random.random()<cond_p:
                    self.theta = new_theta
                    self.prob = new_prob
        weight = 1.0/self.prob
        self.count += 1
        print self.theta
        return (self.theta, weight)

    def __iter__(self):
        return self


def induction(theta, g, funcs, weights, path_num):
    #likelyhood = L(theta|g) = p(g|theta) = the possibility of getting a particular g given parameters theta
    #funcs is a list of ways to make inductions
    #weights is weights for each way of inductions
    p = zeros(g.shape)
    seed = vector2adj(theta, g)
    for f, w in zip(funcs, weights):
        p += w*f(seed)
    diag = eye(p.shape[0])>0
    p[diag] = 0
    p = 2.0*p/sum(p) if sum(p) else p
    absent_prob = ((p*(-1.0)+1.0)**path_num)
    present_sel = logical_and(g, tril(ones(g.shape), 0))
    present_prob_g = absent_prob[present_sel]*(-1.0)+1.0
    absent_sel = logical_and(logical_not(g), tril(ones(g.shape), 0))
    absent_prob_g = absent_prob[absent_sel]
    prob_g = prod(present_prob_g)*prod(absent_prob_g)
    assert prob_g
    return prob_g


def bayesian(sampler, likelyhood_func, prior_func):
    posterior = zeros(sampler.theta.shape)
    p_total = 0
    for theta, weight in sampler:
        l = likelyhood_func(theta)*weight
        prior = prior_func(theta)
        p = l*prior
        posterior += theta*p
        p_total += p
    return posterior/p_total
    

class seed_generator(object):

    def __init__(self, n):
        self.n = n
        self.index = 0

    def next(self):
        #generate a n*n adjacency matrix which corresponds to a undirected graph with no selfloops
        if self.index >= 2**(self.n*(self.n-1)/2):
            raise StopIteration
        s = self.index
        seed = zeros((self.n, self.n))
        for i in arange(self.n):
            for j in arange(i+1, self.n):
                if s:
                    seed[i][j] = s%2
                    seed[j][i] = s%2
                    s = s/2
        print self.index
        self.index = self.index + 1
        return seed

    def __iter__(self):
        return self
    
    
if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt

    g = nx.Graph()
    g.add_edges_from([[1, 2], [1, 3], [2, 4], [3, 4], [1, 4], [1, 5], [5, 6]])
    nx.draw(g)
    plt.show()

    adj = nx.adjacency_matrix(g)
    adj = array(adj, float)
    print adj

    def func1(m):
        return ones(m.shape)

    def func2(m):
        return m

    def func3(m):
        a = dot(m, m)
        for i in arange(a.shape[0]):
            a[i][i] = 0
        return a

    induc_funcs = [func1, func2, func3]
    induc_weights = [0.1, 0.8, 0.5]

    likelyhood_func = lambda theta: induction(theta, adj, induc_funcs, induc_weights, g.number_of_edges())
    prior_func = lambda theta: 1
    sample_dist = lambda theta: differential(likelyhood_func, theta)
    theta = adj2vector(adj)
    sampler = gibbs_sampler(theta, sample_dist)

    posterior = bayesian(sampler, likelyhood_func, prior_func)
    print posterior
    print vector2adj(posterior, adj)
