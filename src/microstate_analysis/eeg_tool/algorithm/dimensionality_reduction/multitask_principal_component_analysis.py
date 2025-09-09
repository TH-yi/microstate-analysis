import autograd.numpy as np
from pymanopt.manifolds import Grassmann, Product
from pymanopt import Problem, solvers
from pymanopt.solvers import SteepestDescent


def zero_mean(data):
    """

    :type data: d-by-n matrices, where d is the number of features and n is the number of instances.
    """
    mean = data.mean(axis=0, keepdims=True)
    return data - mean


def multitask_pca(data, reg, n_k, c_v):
    u_t = []
    score = []
    if c_v:
        estimate_cv(data, reg, n_k)
    else:
        for lamda in reg:
            u_t_temp, data_cov = estimate_fixed_reg(data, lamda, n_k)
            score_temp = eval_score(data_cov, u_t_temp)
            u_t.append(u_t_temp)
            score.append(score_temp)

    return u_t, score

def estimate_cv(data, n_k):
    pass


def estimate_fixed_reg(data, reg, n_k):
    """

    :type n_k: the dimensionality of the principal subspace to be extracted
    :type data: dict{'the name of a task': 'd-by-n matrices of the task'}
    """
    data_all = np.asarray([])
    data_cov = []
    data_d = []
    u_task_t = []
    T = len(data)

    for t in range(0, T):
        if data_all.size == 0:
            data_all = data[t]
        else:
            data_all = np.concatenate((data_all, data[t]),axis=1)
        data_cov.append(np.cov(data[t]))
        data_d.append(data[t].shape[0])

    if reg == 'Inf':
        eigenvalue = np.linalg.eig(np.cov(data_all))
        for t in range(0, T):
            u_task_t[t] = eigenvalue

    elif reg == 'independent':
        for t in range(0, T):
            u_task_t.append(np.linalg.eig(data_cov[t]))

    else:
        u_task_init = []
        for t in range(0, T):
            u_task_init.append(np.eye(data_d[0], n_k))
        u_task_t = multitask_pca_core(data_cov, reg, u_task_init, n_k, data_d[0])
        return u_task_t, data_cov


def multitask_pca_core(covs, lamda, u_task_init, n_k, n_d):
    def cost(x):
        acc_u = 0
        acc_reg = 0
        for i in range(0, len(x)):
            acc_u = acc_u + np.trace(np.linalg.multi_dot([x[i].transpose(), covs[i], x[i]]))
            regu = 0
            for j in range(0, len(x)):
                if i != j:
                    regu = regu + np.trace(np.linalg.multi_dot([x[i], x[i].transpose(), x[j], x[j].transpose()]))
            acc_reg = acc_reg + regu
        f = -0.5*acc_u - (lamda/4)*acc_reg
        return f

    def egrad(x):
        g_u_res = []
        for i in range(0, len(x)):
            g_u = covs[i].dot(x[i])
            for j in range(0, len(x)):
                if i != j:
                    g_u = g_u + lamda * np.linalg.multi_dot([x[j], x[j].transpose(), x[i]])
            g_u_res.append(-g_u)
        return g_u_res

    manifold = Product([Grassmann(n_d, n_k) for t in range(0,len(covs))])
    problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=2)
    solver = solvers.SteepestDescent(maxiter=10000)
    u = solver.solve(problem, x=u_task_init)
    return u

def eval_score(covs, subspace):
    score = []
    for i in range(0, len(covs)):
        score.append(np.trace(np.linalg.multi_dot([subspace[i].transpose(), covs[i], subspace[i]]))/np.trace(covs[i]))
    return score

def generate_data():
    res = []
    for i in range(0, 10):
        res.append(np.random.rand(63, 500))
    return res



if __name__ == '__main__':
    data = generate_data()
    for t in range(0, len(data)):
        data[t] = zero_mean(data[t])

    lamda = np.logspace(-2, 1, 5)
    u_t, score = multitask_pca(data, lamda, 5, False)
    print(score)
