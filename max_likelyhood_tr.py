from numpy import *
from scipy.optimize import minimize, leastsq

import networkx as nx

def adj2pos(g):
    pos = logical_and(g>0, triu(ones(g.shape), 1))
    return pos

def adj2vector(g, pos=None):
    if pos is None:
        pos = adj2pos(g)
    return g[pos]

def vector2adj(x, pos):
    adj = zeros(pos.shape)
    adj[pos] = x
    adj[transpose(pos)] = transpose(adj)[transpose(pos)]
    return adj

def predict(x, adj, coef, xpos):
    def multiply_noselfloop(x, y):
        prod = dot(x, y)
        i = eye(prod.shape[0], dtype=bool)
        prod[i] = 0
        return prod
    def power_noselfloop(x, a):
        if a == 1:
            x[eye(x.shape[0], dtype=bool)] = 0
            return x
        else:
            return multiply_noselfloop(power_noselfloop(x, a-1), x)
    ret = zeros(adj.shape)
    adjone = zeros(adj.shape)
    adjone[xpos] = 1
    adjx = vector2adj(x, xpos)
    coef_norm = sum(coef.values()) * 1.0
    for k in coef:
        pow_k = power_noselfloop(adjx, k)
        pow_norm = sum(power_noselfloop(adjone, k))
        ret += coef[k]/coef_norm * pow_k / pow_norm
    return ret

def residual(x, adj, coef, xpos, ypos):
    pre = predict(x, adj, coef, xpos)
    y_pre = adj2vector(pre, ypos)
    s = sum(adj)
    y = 1.0 * adj2vector(adj, ypos) / s
    r = y - y_pre
    return r

if __name__ == '__main__':
    g = nx.Graph()
    g.add_edges_from([(1, 2), (2, 3), (3, 1), (1, 4), (4, 5), (1, 5), (4, 6), (5, 7)])
    adj = nx.adjacency_matrix(g)
    adj = array(adj.todense(), dtype=float)
    print adj
    x0 = adj2vector(adj)
    x0 = 0.8 * x0 / max(x0)
    print x0
    coef = {1 : 0.5,
            2 : 0.5}
    xpos = adj2pos(adj)
    y0 = predict(x0, adj, coef, xpos)
    ypos = adj2pos(y0)
    '''leastsq
    obj_func = lambda x: residual(x, adj, coef, xpos, ypos)
    opt = leastsq(obj_func, x0)
    succ = opt[1]
    weights = opt[0]
    print 'succ:', succ
    print vector2adj(weights, xpos)
    '''
    obj_func = lambda x: sum(residual(x, adj, coef, xpos, ypos)**2)
    opt = minimize(obj_func, x0, method='TNC', bounds = [(0.0, 1.0) for i in x0], options = {'maxiter' : 1000})
    print opt
    
