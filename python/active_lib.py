#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dolfin import *
from petsc4py import PETSc
import numpy as np
from python.KiriE import silence_everything

# ---------------------------- Helper: sample a control on Q ----------------------------
def sample_m(Q, lengthscale=0.2, sigma=1.0, seed=None):
    """
    Draw a Gaussian-process sample on the DOFs of space Q.
    """
    if seed is not None:
        np.random.seed(seed)

    m = Function(Q, name="control")

    # Squared exponential kernel on R^2
    def se_kernel(x, y, ell, s):
        # x: (Nx, d), y: (Ny, d)
        sqdist = np.sum((x[:, None, :] - y[None, :, :])**2, axis=-1)
        return (s**2) * np.exp(-0.5 * sqdist / (ell**2))

    # DOF coordinates (reshape for 2019.1.0)
    coords = Q.tabulate_dof_coordinates()
    coords = coords.reshape((-1, Q.mesh().geometry().dim()))

    K = se_kernel(coords, coords, lengthscale, sigma)
    K += 1e-8 * np.eye(K.shape[0])  # jitter

    gp = np.random.multivariate_normal(mean=np.zeros(len(coords)), cov=K)

    m_vec = m.vector()
    m_vec.set_local(gp)
    m_vec.apply("insert")
    return m

def make_mass_matrix(Q):
    """Assemble the L2 mass matrix on FunctionSpace Q."""
    fT, q = TrialFunction(Q), TestFunction(Q)
    return assemble(inner(fT, q) * dx)


def norm_list(Flist):
    """
    Compute vector of L2 inner norms.

    Parameters
    ----------
    Flist : list of dolfin.Function

    Returns
    -------
    n : np.ndarray
        Vector of norms.
    """
    if len(Flist) == 0:
        return np.zeros(0)

    Q = Flist[0].function_space()
    mass_matrix = make_mass_matrix(Q)

    vecs = [f.vector() for f in Flist]

    B = len(Flist)
    n = np.zeros(B)

    # Workspace vector for MatVec
    wj = vecs[0].copy()

    for j in range(B):
        wj.zero()
        mass_matrix.mult(vecs[j], wj)
        n[j] = vecs[j].inner(wj)

    return n

def gram_matrix(Flist1, Flist2=None):
    """
    Compute matrix of L2 inner products.

    If Flist2 is None:
        Returns Gram matrix G[i,j] = ∫ f_i f_j dx for Flist1.
    If Flist2 is given:
        Returns matrix G[i,j] = ∫ f1_i f2_j dx.

    Parameters
    ----------
    Flist1 : list of dolfin.Function
    Flist2 : list of dolfin.Function, optional

    Returns
    -------
    G : np.ndarray
        Array of inner products.
    """
    if len(Flist1) == 0:
        return np.zeros((0, 0))

    if Flist2 is None:
        Flist2 = Flist1


    Q = Flist1[0].function_space()
    mass_matrix = make_mass_matrix(Q)

    vecs1 = [f.vector() for f in Flist1]
    vecs2 = [f.vector() for f in Flist2]

    B1, B2 = len(Flist1), len(Flist2)
    G = np.zeros((B1, B2))

    # Workspace vector for MatVec
    wj = vecs2[0].copy()

    for j in range(B2):
        wj.zero()
        mass_matrix.mult(vecs2[j], wj)
        for i in range(B1):
            G[i, j] = vecs1[i].inner(wj)

    return G

def dist_matrix(Flist):
    """
    Compute squared L2 distances D[i,j] = ||f_i - f_j||^2
    using the Gram matrix.

    Returns
    -------
    D : np.ndarray, shape (B,B)
        Squared L2 distance matrix.
    """
    G = gram_matrix(Flist)
    B = G.shape[0]
    D = np.zeros((B, B))

    for i in range(B):
        for j in range(B):
            D[i, j] = G[i, i] + G[j, j] - 2.0 * G[i, j]

    return D


def get_eigenfuncs(GAMMA, G, eps = 1e-8):
    #print("Not scaling by eigvals, as should be!")
    B = len(G)
    assert GAMMA.shape[0]==GAMMA.shape[1]==B

    ed = list(np.linalg.eigh(GAMMA))
    assert np.min(ed[0]) > -1e-6
    ed[0] = np.maximum(ed[0], 0.)
    eigvals, eigvecs = ed

    E = []
    B = len(G)

    for k in range(B):
        #coeffs = eigvecs[:, k]  # eigenvector (length B)
        coeffs = eigvecs[:, k]/(np.sqrt(eigvals[k])+eps)  # eigenvector (length B)
        # New function as linear combination of G[b]
        g_lin = Function(G[0].function_space(), name=f"eigenfunc{k}")
        # Start from zero vector
        g_lin.vector().zero()
        for b in range(B):
            g_lin.vector().axpy(coeffs[b], G[b].vector())  # g_lin += coeffs[b] * G[b]

        g_lin.vector().apply("insert")  # finalize assembly
        E.append(g_lin)

    return E, ed

def linear_combination(F, coef):
    coef = np.array(coef)
    g_lin = Function(F[0].function_space())
    g_lin.vector().zero()
    for i in range(coef.shape[0]):
        g_lin.vector().axpy(coef[i], F[i].vector())  # g_lin += coef[i] * F[i]
    g_lin.vector().apply("insert")  # finalize assembly
    return g_lin

#def linear_combination(F, coef, target_space=None):
#    assert len(F) == len(coef)
#    V = target_space or F[0].function_space()
#    out = Function(V); out.vector().zero()
#    for a, f in zip(coef, F):
#        fv = f if f.function_space().id() == V.id() else project(f, V)
#        out.vector().axpy(float(a), fv.vector())
#    out.vector().apply("add")  # finalize sums
#    return out




class SampleFrom(UserExpression):
    """Wrap a dolfin.Function so it can be interpolated onto another mesh."""
    def __init__(self, f, **kwargs):
        self.f = f                      # set first!
        super().__init__(**kwargs)
    def eval(self, values, x):
        # Evaluate source function at physical point x
        self.f.eval(values, x)
    def value_shape(self):
        return self.f.value_shape()

def pull_to_mesh(q_src, Q_tgt, enforce_bcs=None, expr_degree=4):
    with silence_everything():
        # Fast path: same mesh object
        if q_src.function_space().mesh().id() == Q_tgt.mesh().id():
            q_tgt = Function(Q_tgt)
            q_tgt.assign(q_src)
            if enforce_bcs:
                for bc in enforce_bcs: bc.apply(q_tgt.vector())
            return q_tgt

        # Split source into components
        u_src, beta_src = q_src.split(deepcopy=True)  # [u(3), beta(2)]

        # Target component spaces (respect enrichment)
        U_tgt = Q_tgt.sub(0).collapse()  # enriched [P2+B3]^3
        B_tgt = Q_tgt.sub(1).collapse()  # [P2]^2

        # Wrap source as expressions; pass the target element to avoid value_shape calls
        u_expr    = SampleFrom(u_src,  element=U_tgt.ufl_element(),  degree=expr_degree)
        beta_expr = SampleFrom(beta_src, element=B_tgt.ufl_element(), degree=expr_degree)

        # Interpolate onto target component spaces
        u_t = interpolate(u_expr, U_tgt)
        b_t = interpolate(beta_expr, B_tgt)

        # Pack into mixed
        q_tgt = Function(Q_tgt)
        FunctionAssigner(Q_tgt, [U_tgt, B_tgt]).assign(q_tgt, [u_t, b_t])

        if enforce_bcs:
            for bc in enforce_bcs: bc.apply(q_tgt.vector())
        return q_tgt

