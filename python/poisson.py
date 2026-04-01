from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

class PoissonControl(object):

    def __init__(self, nx=32, ny=32, alpha=1e-3, mesh = None):
        self.alpha = alpha

        # Mesh & spaces
        if mesh is None:
            self.mesh = UnitSquareMesh(nx, ny)
        else:
            self.mesh = mesh
        self.V = FunctionSpace(self.mesh, "CG", 1)  # state / adjoint
        self.Q = FunctionSpace(self.mesh, "CG", 1)  # control

        # Homogeneous Dirichlet everywhere
        self.bc = DirichletBC(self.V, Constant(0.0), "on_boundary")

        # Target field u_d(x,y) = sin(4πx) sin(πy)
        self.u_d = interpolate(Expression("sin(4.0*pi*x[0]) * sin(pi*x[1])",
                                          degree=4, pi=np.pi),
                               self.V)

        # Functions to hold solutions
        self.u   = Function(self.V, name="state")
        self.lam = Function(self.V, name="adjoint")

        # Trial/test
        uT, v = TrialFunction(self.V), TestFunction(self.V)
        lT, w = TrialFunction(self.V), TestFunction(self.V)
        gT, q = TrialFunction(self.Q), TestFunction(self.Q)

        # System matrices that do NOT depend on m (reuse every call)
        self.a_state = inner(grad(uT),  grad(v)) * dx
        self.A_state = assemble(self.a_state)
        self.bc.apply(self.A_state)

        self.a_adj   = inner(grad(lT),  grad(w)) * dx
        self.A_adj   = assemble(self.a_adj)
        self.bc.apply(self.A_adj)

        # L2 projection (mass) matrix on Q for returning gradient as a Function(Q)
        self.a_proj  = inner(gT, q) * dx
        self.A_proj  = assemble(self.a_proj)

        # Reusable RHS vectors
        self.b_state = None
        self.b_adj   = None
        self.b_proj  = None

        # Reusable test functions for RHS assembly
        self.v = v
        self.w = w
        self.q = q

    def evaluate(self, m):
        """
        Given control m (Function on self.Q), compute:
          - state u, adjoint lam
          - J(m)
          - gradient g in self.Q (via mass-matrix projection)
        Returns: (J_val, g)
        """
        # --- State: (∇u, ∇v) = (m, v)
        L_state = inner(m, self.v) * dx
        self.b_state = assemble(L_state)
        self.bc.apply(self.b_state)
        solve(self.A_state, self.u.vector(), self.b_state)

        # --- Adjoint: (∇λ, ∇w) = (u - u_d, w)
        L_adj = inner(self.u - self.u_d, self.w) * dx
        self.b_adj = assemble(L_adj)
        self.bc.apply(self.b_adj)
        solve(self.A_adj, self.lam.vector(), self.b_adj)

        # --- Cost value
        J_form = 0.5*inner(self.u - self.u_d, self.u - self.u_d)*dx \
               + 0.5*self.alpha*inner(m, m)*dx
        J_val = assemble(J_form)

        # --- Gradient in Q:
        L_proj = self.alpha*inner(m, self.q)*dx - inner(self.lam, self.q)*dx
        self.b_proj = assemble(L_proj)
        g = Function(self.Q, name="gradJ")
        solve(self.A_proj, g.vector(), self.b_proj)

        return J_val, g

    def plot(self, m, title="", show=True, fname='temp.png'):
        # Coerce m into the control space Q on this mesh
        mQ = None
        if isinstance(m, Function):
            # If it's on a different space/mesh or different element, (re)interpolate to Q
            try:
                same_mesh = (m.function_space().mesh().id() == self.Q.mesh().id())
            except AttributeError:
                same_mesh = False
            if same_mesh and m.function_space().ufl_element().family() == self.Q.ufl_element().family() \
               and m.function_space().ufl_element().degree() == self.Q.ufl_element().degree():
                mQ = m
            else:
                mQ = interpolate(m, self.Q)
        else:
            # Expression/Constant or similar → interpolate into Q
            mQ = interpolate(m, self.Q)

        # Do the actual plot
        plt.figure(figsize=[4,4])
        c = plot(mQ)  # FEniCS' built-in plot
        #plt.colorbar(c)
        plt.title(title, fontdict = {'weight':'bold'})
        plt.tight_layout()
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

if __name__ == '__main__':
    from python.active_lib import sample_m
    co = PoissonControl(nx=32, ny=32, alpha=1e-3)
    np.random.seed(123)
    m = sample_m(co.Q)
    _, g = co.evaluate(m)
    co.plot(m, title=r'', fname = 'poisson_m.pdf')
    co.plot(g, title=r'', fname = 'poisson_g.pdf')
