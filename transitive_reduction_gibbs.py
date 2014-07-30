from copy import deepcopy

from numpy import *


def toggle(theta, i):
    #toggle ith element of theta
    new_theta = deepcopy(theta)
    new_theta[i] = float(not(new_theta[i]))
    return new_theta


def vector2adj(theta, g):
    sel = logical_and(g>0, triu(ones(g.shape), 1))
    adj = zeros(g.shape)
    adj[sel] = theta
    adj[transpose(sel)] = transpose(adj)[transpose(sel)]
    return adj


def adj2vector(g):
    sel = logical_and(g>0, triu(ones(g.shape), 1))
    return g[sel]
                      
    
def induction(theta, g, funcs, weights, path_num):
    #likelyhood = L(theta|g) = p(g|theta) = the possibility of getting a particular g given parameters theta
    #funcs is a list of ways to make inductions
    #weights is weights for each way of inductions
    p = zeros(g.shape)
    seed = vector2adj(theta, g)
    weights = array(weights, float)
    weights = weights/sum(weights)
    for f, w in zip(funcs, weights):
        p += w*f(seed)
    diag = eye(p.shape[0])>0
    p[diag] = 0
    absent_prob = ((p*(-1.0)+1.0)**path_num)
    present_sel = logical_and(g, tril(ones(g.shape), -1))
    present_prob_g = absent_prob[present_sel]*(-1.0)+1.0
    absent_sel = logical_and(logical_not(g), tril(ones(g.shape), -1))
    absent_prob_g = absent_prob[absent_sel]
    prob_g = prod(present_prob_g)*prod(absent_prob_g)
    assert prob_g
    return prob_g


def differential(func, theta):
    #calculates product of differentials of func over theta in all dimensions
    value = func(theta)
    #print value
    dif = 1.0
    for i in range(0, len(theta)):
        new_theta = toggle(theta, i)
        d_theta = new_theta[i] - theta[i]
        d_value = func(new_theta)-value
        slope = d_value/d_theta
        #print slope
        assert slope
        dif = dif*slope
    return abs(dif)**(1.0/len(theta))   #I can't really explain why root of dif is used. it's the average diff in one dimension. i thought the product of diffs in all dimension should be correlated with my notion of weight. but in that case the sampler moves in a narrow space because in certain dimensions one value is so much higher prefered than the other.

        
class gibbs_sampler(object):

    def __init__(self, theta_0, dist, ignore = 100, total = 1000):
        self.theta = theta_0   #initial theta
        self.dist = dist   #desired distribution function, dist(theta) is proportional to the desired possibility of picking theta
        self.prob = self.dist(self.theta)   #possibility of sampling theta, inverse of its weight
        self.total = total+ignore
        self.count = 0
        for i in range(0, ignore):
            self.next()
        print 'gibbs_sampler initiated'

    def next(self):
        if self.count >= self.total:
            raise StopIteration
        for j in range(0, 10):
            for i in range(0, len(self.theta)):
                new_theta = toggle(self.theta, i)
                new_prob = self.dist(new_theta)
                cond_p = new_prob/(new_prob+self.prob)
                if random.random()<cond_p:
                    self.theta = new_theta
                    self.prob = new_prob
        weight = 1.0
        self.count += 1
        print self.theta, weight
        return (self.theta, weight)

    def __iter__(self):
        return self


def bayesian(sampler, likelyhood_func, prior_func):
    theta_sum = zeros(sampler.theta.shape)
    total = 0
    weighted_total = 0
    posterior = zeros(sampler.theta.shape)
    p_total = 0
    for theta, weight in sampler:
        l = likelyhood_func(theta)*weight
        prior = prior_func(theta)
        p = l*prior
        theta_sum += theta
        total += 1
        weighted_total += theta*weight
        posterior += theta*p
        p_total += p
    print theta_sum/total
    print weighted_total/total
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
    g.add_edges_from([[1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 3], [4, 5], [5, 6]])
    nx.draw(g)
    plt.show()

    adj = nx.adjacency_matrix(g)
    adj = adj.toarray()
    adj = array(adj, float)
    print adj

    def func1(m):
        p = ones(m.shape)
        p = 2.0*p/sum(p) if sum(p) else p
        return p

    def func2(m):
        return 2.0*m/sum(m) if sum(m) else m

    def func3(m):
        a = dot(m, m)
        a[eye(a.shape[0])>0] = 0
        return 2.0*a/sum(a) if sum(a) else a

    induc_funcs = [func1, func2, func3]
    induc_weights = [0.1, 0.8, 0.8]

    likelyhood_func = lambda theta: induction(theta, adj, induc_funcs, induc_weights, g.number_of_edges())
    prior_func = lambda theta: 1
    #sample_dist = lambda theta: differential(likelyhood_func, theta)
    sample_dist = likelyhood_func
    theta = adj2vector(adj)
    theta = array(random.rand(len(theta))>0.2, float)
    sampler = gibbs_sampler(theta, sample_dist)

    posterior = bayesian(sampler, likelyhood_func, prior_func)
    print posterior
    print vector2adj(posterior, adj)
