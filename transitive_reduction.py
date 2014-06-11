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
    
    
def induction(s, weights, funcs):
    #predict possibility distribution of possible adj matrix induced from a seed matrix
    #m is the seed matrix
    #weights is weights for each way of inductions
    #funcs is a list of ways to make inductions
    p = zeros(s.shape)
    for f, w in zip(funcs, weights):
        p += w*f(s)
    for i in arange(p.shape[0]):
        p[i][i] = 0
    p = 2.0*p/sum(p)
    return p


def bayesian(m, weights, funcs):
    #calculates how likely each edge in m is likely to be in the transitive reduction of m
    edge_num = m.sum()/2
    n = m.shape[0]
    seeds = seed_generator(n)

    p_total = 0
    p = zeros(m.shape)
    for s in seeds:
        p_s = induction(s, weights, funcs)
        p_absent = (1+(-1.0)*p_s)**edge_num
        p_present = 1+(-1.0)*p_absent
        p_m = 1.0
        for i in arange(n):
            for j in arange(i+1, n):
                if m[i][j]:
                    p_m *= p_present[i][j]
                else:
                    p_m *= p_absent[i][j]
        p_total += p_m
        p += p_m*s
    return p/p_total
    

if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt

    g = nx.Graph()
    g.add_edges_from([[1, 2], [1, 3], [2, 4], [3, 4], [1, 4], [1, 5], [5, 6], [6, 7]])
    nx.draw(g)
    plt.show()

    m = nx.adjacency_matrix(g)
    m = array(m)
    print m

    def func1(m):
        return ones(m.shape)

    def func2(m):
        return m

    def func3(m):
        a = dot(m, m)
        for i in arange(a.shape[0]):
            a[i][i] = 0
        return a

    funcs = [func1, func2, func3]
    weights = [0.1, 0.8, 0.5]

    baye = bayesian(m, weights, funcs)
    print baye
