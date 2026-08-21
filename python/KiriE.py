# -*- coding: utf-8 -*-

import os, numpy as np
from dolfin import *
from ufl import Index, SpatialCoordinate, as_vector
import matplotlib.pyplot as plt

set_log_level(LogLevel.ERROR)
from ffc import log as ffc_log
ffc_log.set_level(ffc_log.ERROR)

import os, sys, contextlib

@contextlib.contextmanager
def silence_everything():
    # flush Python streams first
    sys.stdout.flush(); sys.stderr.flush()
    # save original fds
    old_out, old_err = os.dup(1), os.dup(2)
    # open devnull
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        # redirect to devnull
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        # restore
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out); os.close(old_err); os.close(devnull)

class ShellEnergy:
    """
    Nonlinear Naghdi shell for classic FEniCS (dolfin 2019.1.x).

    New workflow:
      - sample_m(...) -> Function(U)     # displacement only
      - evaluate(u, P, ...)              # internally solves beta* for this u,
                                         # builds q=(u, beta*), then computes energy & gradient.
    """

    def __init__(self,
                 xdmf_path=None,
                 mesh=None,
                 use_spiral_perim=False,
                 E=2.0e9/1e2,     # Young's modulus (converted to cm units)
                 nu=0.22,         # Poisson
                 t_val=1e-4,      # thickness [cm]
                 quadrature_full=4,
                 quadrature_reduced=2):
        with silence_everything():
            # --- Mesh ---
            if mesh is None:
                if xdmf_path is None:
                    raise ValueError("Provide either a mesh or an xdmf_path")
                self.mesh = Mesh()
                with XDMFFile(xdmf_path) as infile:
                    infile.read(self.mesh)
            else:
                self.mesh = mesh

            parameters["form_compiler"]["quadrature_degree"] = quadrature_full
            self.dx   = Measure("dx", domain=self.mesh, metadata={"quadrature_degree": quadrature_full})
            self.dx_h = Measure("dx", domain=self.mesh, metadata={"quadrature_degree": quadrature_reduced})
            self.X      = SpatialCoordinate(self.mesh)

            # --- Material & constants (cm-based units) ---
            self.E   = Constant(E)
            self.nu  = Constant(nu)
            self.mu  = self.E/(2.0*(1.0 + self.nu))
            self.lmb = 2.0*self.mu*self.nu/(1.0 - 2.0*self.nu)
            self.t   = Constant(t_val)

            # gravity/material densities used in your external work pieces
            self.rho_g = Constant((1200-1000)*9.81 / ((1e2)**2))         # cm-units
            rho_org    = (1080-1000)*9.81/(1e2*1e2)                      # cm-units
            self.mg_org_per_column = Constant(rho_org*2*0.06)            # matches your script

            # --- Spaces: [ (P2 + bubble)^3 ] x [ (P2)^2 ] ---
            P2      = FiniteElement("Lagrange", self.mesh.ufl_cell(), degree=2)
            B3      = FiniteElement("B",        self.mesh.ufl_cell(), degree=3)
            enr     = P2 + B3
            element = MixedElement([VectorElement(enr, dim=3), VectorElement(P2, dim=2)])
            self.Q  = FunctionSpace(self.mesh, element)

            # Helper spaces
            self.V_phi    = FunctionSpace(self.mesh, VectorElement("P", self.mesh.ufl_cell(), degree=2, dim=3))
            self.V_normal = FunctionSpace(self.mesh, VectorElement("P", self.mesh.ufl_cell(), degree=1, dim=3))
            self.W1       = FunctionSpace(self.mesh, 'CG', 1)

            # --- Initial mid-surface and director (flat plate by default) ---
            self.phi0 = project(Expression(('x[0]','x[1]','0'), degree=4), self.V_phi)

            def normal(y):
                n = cross(y.dx(0), y.dx(1))
                return n/sqrt(inner(n,n))

            self.d0 = project(normal(self.phi0), self.V_normal)

            # angles beta0 s.t. director(d(beta0)) = d0
            beta0_expr = Expression(["atan2(-n[1], sqrt(pow(n[0],2) + pow(n[2],2)))",
                                     "atan2(n[0],n[2])"], n=self.d0, degree=4)
            V_beta = FunctionSpace(self.mesh, VectorElement("P", self.mesh.ufl_cell(), degree=2, dim=2))
            self.beta0 = project(beta0_expr, V_beta)

            # --- Geometry tensors on reference surface ---
            self.a0    = grad(self.phi0).T*grad(self.phi0)
            self.a0inv = inv(self.a0)
            self.b0    = -0.5*(grad(self.phi0).T*grad(self.d0) + grad(self.d0).T*grad(self.phi0))
            self.j0    = det(self.a0)

            # --- Plane-stress contravariant 4th-order tensor A_[i,j,l,m] ---
            i,j,l,m = Index(), Index(), Index(), Index()
            self.A = as_tensor((((2.0*self.lmb*self.mu)/(self.lmb + 2.0*self.mu))*self.a0inv[i,j]*self.a0inv[l,m]
                               + 1.0*self.mu*(self.a0inv[i,l]*self.a0inv[j,m] + self.a0inv[i,m]*self.a0inv[j,l])),
                               [i,j,l,m])

            # --- PSRI controls ---
            h          = CellDiameter(self.mesh)
            self.alpha = project(self.t**2/h**2, FunctionSpace(self.mesh, 'DG', 0))

            # --- Boundary conditions ---
            if use_spiral_perim:
                # matches your "perim" logic; clamps all 5 DOFs on that shape
                def perim(x, on_boundary):
                    r = np.sqrt(x[0]**2 + x[1]**2)
                    theta = np.arctan2(x[1], x[0])
                    return (r > (0.5 - 0.055)*(1.0 + 0.05*np.sin((theta - np.deg2rad(30))*16))) and on_boundary
            else:
                # fallback: clamp the *geometric* boundary
                def perim(x, on_boundary):
                    return on_boundary

            self.zero_q = Constant((0.0,0.0,0.0, 0.0,0.0))
            self.perim_cb = perim
            self.bcs    = [DirichletBC(self.Q, self.zero_q, perim)]

            # --- H1-like block inner product for gradient smoothing (metric M) ---
            u_tr, b_tr = TrialFunctions(self.Q)   # (u, beta) trials
            v_u,  v_b  = TestFunctions(self.Q)    # (u, beta) tests

            Lchar = Constant(self.mesh.hmax()**2)   # characteristic length^2
            #grad_pen = 1e1
            #grad_pen = 1e2
            grad_pen = 1e4
            kappa_u = Constant(grad_pen) * Lchar
            wbeta_L2 = Constant(1.0)
            wbeta_H1 = Constant(grad_pen) * Lchar

            m_u = ( inner(u_tr, v_u) + kappa_u * inner(grad(u_tr), grad(v_u)) ) * self.dx
            m_b = ( wbeta_L2 * inner(b_tr, v_b) + wbeta_H1 * inner(grad(b_tr), grad(v_b)) ) * self.dx
            self.m_form = m_u + m_b

            # Assemble and apply BCs
            self.M = assemble(self.m_form)
            with silence_everything():
                for bc in self.bcs:
                    bc.apply(self.M)

            self.M_solver = LUSolver(self.M)

        u_prox = self.sample_m(seed=123)
        #u_prox = self.sample_m(seed=123, ell_u = 2.)
        self.set_reference_from_u(u_prox)
        self.plot(u_prox, fname = 'u_prox_init.png')

    # --- Director mapping from angles (2 -> 3) ---
    @staticmethod
    def director(beta_vec):
        # beta_vec is a 2-vector field: [beta0, beta1]
        return as_vector([sin(beta_vec[1])*cos(beta_vec[0]),
                          -sin(beta_vec[0]),
                          cos(beta_vec[1])*cos(beta_vec[0])])

    # --- Kinematics-dependent strain measures (built on the fly for supplied q_) ---
    def _strains(self, u_, beta_):
        F     = grad(u_) + grad(self.phi0)
        d     = ShellEnergy.director(beta_ + self.beta0)
        e     = 0.5*(F.T*F - self.a0)                                   # membrane strain
        kappa = -0.5*(F.T*grad(d) + grad(d).T*F) - self.b0              # bending strain
        gamma = F.T*d - grad(self.phi0).T*self.d0                       # shear strain
        return F, d, e, kappa, gamma

    # --- Elastic energy functional Π (PSRI split) for a given q_ ---
    def _elastic_energy(self, q_):
        u_, beta_ = split(q_)
        F, d, e, kappa, gamma = self._strains(u_, beta_)

        i,j,l,m = Index(), Index(), Index(), Index()
        N = as_tensor(self.t * self.A[i,j,l,m]*e[l,m], [i,j])
        M = as_tensor((self.t**3/12.0) * self.A[i,j,l,m]*kappa[l,m], [i,j])
        T = as_tensor(self.t * self.mu * self.a0inv[i,j]*gamma[j], [i])

        psi_m = 0.5*inner(N, e)
        psi_b = 0.5*inner(M, kappa)
        psi_s = 0.5*inner(T, gamma)

        Pi_PSRI = ( psi_b*sqrt(self.j0)*self.dx
                  + self.alpha*psi_m*sqrt(self.j0)*self.dx
                  + self.alpha*psi_s*sqrt(self.j0)*self.dx
                  + (1.0 - self.alpha)*psi_s*sqrt(self.j0)*self.dx_h
                  + (1.0 - self.alpha)*psi_m*sqrt(self.j0)*self.dx_h )
        return Pi_PSRI

    # --- External work (kept for completeness; not subtracted in energy below, mirroring original use) ---
    def _external_work(self, q_, P):
        u_, beta_ = split(q_)
        # (Your original code overwrote the first term; keep sum commented for optional use)
        weight  = (1 - self.alpha) * inner(Constant((0.0, 0.0, -P*self.rho_g*1e-4)), u_) * self.dx_h
        R = sqrt(self.X[0]**2 + self.X[1]**2)
        weight = (1 - self.alpha) * Constant(-P/5.0) * self.mg_org_per_column \
                  * exp(-(R**4)/(0.06**4)) * u_[2] * self.dx_h
        return weight

    def evaluate(self, m, **kwargs):
        with silence_everything():
            return self._evaluate(m, **kwargs)


    def _assemble_q_from_u(self, u_given,
                           beta_init=None,
                           newton_max_it=25, newton_tol=1e-10):
        # Ensure canonical spaces/assigners exist
        self._ensure_canonical_spaces()

        # Project u into the canonical enriched displacement subspace
        if isinstance(u_given, Function) and u_given.function_space().id() == self.Uspace.id():
            u_enr = u_given
        else:
            u_enr = project(u_given, self.Uspace)

        # Compute optimal beta* for this u
        beta_star = self.optimize_beta(u_enr,
                                       beta_init=beta_init,
                                       newton_max_it=newton_max_it,
                                       newton_tol=newton_tol)

        # Ensure beta lives in the canonical beta subspace
        if not isinstance(beta_star, Function) or beta_star.function_space().id() != self.Bspace.id():
            beta_star = project(beta_star, self.Bspace)

        # Assemble q = (u, beta*) via the mixed assigner (no manual DOF shuffling)
        q = Function(self.Q)
        self.assign_q_from_parts.assign(q, [u_enr, beta_star])

        # Enforce Dirichlet BCs on q
        with silence_everything():
            for bc in self.bcs:
                bc.apply(q.vector())

        return q, u_enr, beta_star


    def _evaluate(self, u, P=5.,
          return_function=True,
          reg_kind='H1', reg_alpha=0.0,
          u_weight=1.0, beta_weight=0.0, surface_weight=True, q_ref=None,
          balanced=False, balance_include_lambda=True, eps=1e-16):
        """
        Implements h(u - u_ref):
          1) form u_shift = u - u_ref  (u_ref from self.q_ref if set, else 0)
          2) solve beta* = argmin_beta Π_elastic(u_shift, beta)
          3) build q=(u_shift, beta*)
          4) compute energy and gradient (regularizer centered at 0 by default)

        Returns:
          (Pi_total_scalar, grad)
        where `grad` is either:
          - Function(self.Q)  : Riesz representer (if return_function=True)
          - PETScVector       : assembled residual vector (if False)
        """
        # Project input u to Uspace
        Uspace = self.Q.sub(0).collapse()
        if isinstance(u, Function) and u.function_space().dim() == Uspace.dim():
            u_proj = u
        else:
            u_proj = project(u, Uspace)

        # Shift by stored reference
        u_ref = self._u_ref_in_U()
        u_shift = project(u_proj - u_ref, Uspace)

        # Build q=(u_shift, beta*) from shifted u
        q_, u_enr, beta_star = self._assemble_q_from_u(
            u_shift, beta_init=None,
            newton_max_it=25, newton_tol=1e-10
        )

        # Base elastic energy
        Pi_el = self._elastic_energy(q_)
        self.Pi_el = Pi_el

        # Regularizer centered at 0 so the functional is of the shift only
        q_zero = Function(self.Q); q_zero.vector().zero(); q_zero.vector().apply("insert")
        reg_form = self.regularizer(q_, q_ref=(q_ref if q_ref is not None else q_zero), kind=reg_kind,
                                    surface_weight=surface_weight,
                                    u_weight=u_weight, beta_weight=beta_weight)
        self.Reg_form = reg_form

        # Total potential used here: elastic + reg (external work omitted by design)
        Pi_tot = Constant(1-reg_alpha)*Pi_el + Constant(reg_alpha) * reg_form

        # Scalar energy
        Pi_scalar = assemble(Pi_tot)

        if not balanced:
            # Residual and metric mapping
            R_form = derivative(Pi_tot, q_, TestFunction(self.Q))
            b = assemble(R_form)
            for bc in self.bcs:
                bc.apply(b)

            if not return_function:
                return float(Pi_scalar), b

            r_fun = Function(self.Q)
            self.M_solver.solve(r_fun.vector(), b)
            u_grad_fun = self._u_grad_from_mixed(r_fun, u)
            smooth = True
            #smooth = False
            if smooth:
                u_grad_fun = self.helmholtz_filter_u_fast(u_grad_fun, 5e-2)
            return float(Pi_scalar), u_grad_fun

        assert False
        # ---------- Balanced mode ----------
        R_el = derivative(Pi_el, q_, TestFunction(self.Q))
        b_el = assemble(R_el);  [bc.apply(b_el) for bc in self.bcs]

        R_reg = derivative(reg_form, q_, TestFunction(self.Q))
        b_reg = assemble(R_reg); [bc.apply(b_reg) for bc in self.bcs]
        b_reg *= reg_alpha

        # Post-metric gradients for scaling
        renorm = "l2"
        r_el = Function(self.Q); self.M_solver.solve(r_el.vector(), b_el)
        nb_el = b_el.norm(renorm); nr_el = r_el.vector().norm(renorm)
        s_el = (1-reg_alpha) * nb_el / (nr_el + eps)

        r_reg = Function(self.Q); self.M_solver.solve(r_reg.vector(), b_reg)
        nb_reg = b_reg.norm(renorm); nr_reg = r_reg.vector().norm(renorm)
        s_reg = reg_alpha * nb_reg / (nr_reg + eps)

        if return_function:
            r_bal = Function(self.Q)
            r_bal.vector().zero()
            r_bal.vector().axpy(s_el,  r_el.vector())
            r_bal.vector().axpy(s_reg, r_reg.vector())
            r_bal.vector().apply("insert")
            return float(Pi_scalar), r_bal
        else:
            b_bal = b_el.copy(); b_bal *= 0.0
            b_bal.axpy(s_el,  b_el)
            b_bal.axpy(s_reg, b_reg)
            return float(Pi_scalar), b_bal

    def build_taper(self, radius=0.05, power=2, normalize=True, mesh=None):
        """
        Build a smooth bump b(x) ∈ CG1 on the given mesh (or self.mesh if None) such that:
            (I - r^2 Δ) b = 1 in Ω,  b = 0 on (clamped) boundary (self.perim_cb).
        Args:
            radius: boundary layer thickness
            power:  sharpness (b^power)
            normalize: scale max(b) ≈ 1
            mesh:   mesh to build the taper on (defaults to self.mesh)
        Returns:
            bfun: Function in CG1 on 'mesh'
        """
        mesh = mesh or self.mesh
        V = FunctionSpace(mesh, 'CG', 1)
        b  = TrialFunction(V)
        v  = TestFunction(V)
        r2 = Constant(radius**2)

        dx_local = Measure("dx", domain=mesh)
        a = (b*v + r2*dot(grad(b), grad(v))) * dx_local
        L = 1.0 * v * dx_local

        # Dirichlet 0 on the SAME boundary rule (self.perim_cb) but for this mesh
        bc = DirichletBC(V, Constant(0.0), self.perim_cb)

        bfun = Function(V)
        solve(a == L, bfun, bc,
              solver_parameters={"linear_solver":"cg", "preconditioner":"hypre_amg"})

        if normalize:
            mx = bfun.vector().max()
            if mx > 0:
                bvec = bfun.vector()
                bvec[:] /= mx
                bvec.apply("insert")

        if power != 1:
            bfun = project(bfun**power, V)

        return bfun

    def taper_scalar(self, f, radius=0.05, power=2):
        """
        Multiply a scalar Function/expr 'f' by the taper on its mesh.
        Returns Function in the same scalar CG space as projection target.
        """
        # choose mesh from f if possible
        if isinstance(f, Function):
            mesh = f.function_space().mesh()
        else:
            mesh = self.mesh
        V = FunctionSpace(mesh, 'CG', 1)
        bump = self.build_taper(radius=radius, power=power, mesh=mesh)
        return project(f * bump, V)


    def taper_vector(self, v, radius=0.05, power=2, out_space=None):
        """
        Multiply a 3D vector Function/expr 'v' by the taper on its mesh.
        Returns Function in 'out_space' if given, else in v's vector space
        (or the enriched Uspace if v is an expression).
        """
        if isinstance(v, Function):
            mesh = v.function_space().mesh()
            Vout = out_space or v.function_space()
        else:
            mesh = self.mesh
            Vout = out_space or self.Q.sub(0).collapse()  # enriched displacement space

        bump = self.build_taper(radius=radius, power=power, mesh=mesh)
        tapered_expr = as_vector((v[0]*bump, v[1]*bump, v[2]*bump))
        return project(tapered_expr, Vout)



    def _u_grad_from_mixed(self, r_mixed, u_in):
        """
        Extract the u-gradient from the mixed Function r_mixed (on self.Q).
        Return it in a space compatible with u_in:
          - if u_in is a Function, return in u_in.function_space()
          - else, return in Uspace (self.Q.sub(0).collapse()).
        """
        # 1) get a proper subfunction on the u-subspace
        u_sub = r_mixed.sub(0, deepcopy=True)   # lives on the subspace of self.Q

        # 2) choose output space
        if isinstance(u_in, Function):
            Vout = u_in.function_space()
        else:
            Vout = self.Q.sub(0).collapse()

        # 3) project into Vout (robust mapping between spaces)
        try:
            u_grad_out = project(u_sub, Vout)
        except Exception:
            # Fallback: go via the collapsed Uspace and then project again
            Uspace = self.Q.sub(0).collapse()
            u_grad_mid = project(u_sub, Uspace)
            u_grad_out = project(u_grad_mid, Vout)

        return u_grad_out


    def diagnose_energy(self, q_):
        # Split fields
        u_, beta_ = split(q_)
        F, d, e, kappa, gamma = self._strains(u_, beta_)

        i,j,l,m = Index(), Index(), Index(), Index()
        N = as_tensor(self.t * self.A[i,j,l,m]*e[l,m], [i,j])
        M = as_tensor((self.t**3/12.0) * self.A[i,j,l,m]*kappa[l,m], [i,j])
        T = as_tensor(self.t * self.mu * self.a0inv[i,j]*gamma[j], [i])

        psi_m = 0.5*inner(N, e)
        psi_b = 0.5*inner(M, kappa)
        psi_s = 0.5*inner(T, gamma)

        # PSRI split: report *each* contribution explicitly
        U_b_full = assemble( psi_b * sqrt(self.j0) * self.dx )

        U_m_full = assemble( self.alpha      * psi_m * sqrt(self.j0) * self.dx )
        U_m_red  = assemble( (1.0 - self.alpha) * psi_m * sqrt(self.j0) * self.dx_h )

        U_s_full = assemble( self.alpha      * psi_s * sqrt(self.j0) * self.dx )
        U_s_red  = assemble( (1.0 - self.alpha) * psi_s * sqrt(self.j0) * self.dx_h )

        U_total  = U_b_full + U_m_full + U_m_red + U_s_full + U_s_red

        # alpha stats (no projection needed; DG0 integrates fine)
        vol    = assemble(Constant(1.0) * self.dx)
        a_int  = assemble(self.alpha * self.dx)
        a_min  = self.alpha.vector().min()
        a_max  = self.alpha.vector().max()
        a_avg  = a_int / vol if vol > 0 else float('nan')

        return {
            "U_total": U_total,
            "U_b_full": U_b_full,
            "U_m_full(alpha*dx)": U_m_full,
            "U_m_red((1-alpha)*dx_h)": U_m_red,
            "U_s_full(alpha*dx)": U_s_full,
            "U_s_red((1-alpha)*dx_h)": U_s_red,
            "alpha_min": float(a_min),
            "alpha_max": float(a_max),
            "alpha_avg": float(a_avg),
        }

    def optimize_beta(self, u_given, beta_init=None,
                      newton_max_it=25, newton_tol=1e-10):
        """
        Find beta* that minimizes elastic energy for a fixed displacement u:
            dΠ_elastic(u, beta)/d beta = 0
        External work W does not depend on beta, so it is omitted.
        """

        # Ensure u is a Function in the enriched displacement subspace
        Uspace = self.Q.sub(0).collapse()
        if isinstance(u_given, Function) and u_given.function_space().dim() == Uspace.dim():
            u_func = u_given
        else:
            u_func = project(u_given, Uspace)

        # Unknown beta in its subspace
        Bspace = self.Q.sub(1).collapse()
        beta   = Function(Bspace)
        if beta_init is not None:
            beta.assign(project(beta_init, Bspace))
        else:
            beta.vector().zero(); beta.vector().apply("insert")  # start from 0

        eta = TestFunction(Bspace)
        db  = TrialFunction(Bspace)

        # Build strains with fixed u and variable beta
        F     = grad(u_func) + grad(self.phi0)
        d     = ShellEnergy.director(beta + self.beta0)
        e     = 0.5*(F.T*F - self.a0)
        kappa = -0.5*(F.T*grad(d) + grad(d).T*F) - self.b0
        gamma = F.T*d - grad(self.phi0).T*self.d0

        i,j,l,m = Index(), Index(), Index(), Index()
        N = as_tensor(self.t * self.A[i,j,l,m]*e[l,m], [i,j])
        M = as_tensor((self.t**3/12.0) * self.A[i,j,l,m]*kappa[l,m], [i,j])
        T = as_tensor(self.t * self.mu * self.a0inv[i,j]*gamma[j], [i])

        psi_m = 0.5*inner(N, e)
        psi_b = 0.5*inner(M, kappa)
        psi_s = 0.5*inner(T, gamma)

        # Elastic energy (PSRI split), identical to the class definition
        Pi_el = ( psi_b*sqrt(self.j0)*self.dx
                + self.alpha*psi_m*sqrt(self.j0)*self.dx
                + self.alpha*psi_s*sqrt(self.j0)*self.dx
                + (1.0 - self.alpha)*psi_s*sqrt(self.j0)*self.dx_h
                + (1.0 - self.alpha)*psi_m*sqrt(self.j0)*self.dx_h )

        # Residual and tangent wrt beta (u is fixed)
        R = derivative(Pi_el, beta, eta)
        J = derivative(R,     beta, db)

        # Same clamped boundary for beta as used in the full problem
        bc_beta = DirichletBC(Bspace, Constant((0.0, 0.0)), self.perim_cb)

        # Nonlinear solve (beta only)
        problem = NonlinearVariationalProblem(R, beta, bcs=[bc_beta], J=J)
        solver  = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm["newton_solver"]["maximum_iterations"] = newton_max_it
        prm["newton_solver"]["absolute_tolerance"] = newton_tol
        prm["newton_solver"]["relative_tolerance"] = newton_tol
        prm["newton_solver"]["linear_solver"]      = "mumps"  # or "lu"/"cg"+"hypre_amg"
        # Optional line search helps if the initial guess is far
        #prm["newton_solver"]["line_search"]        = "bt"

        solver.solve()
        return beta

    #def sample_m(self, ell_u=0.5, amp_u=1e-2, amp_u_tang=None,
    def sample_m(self, ell_u=2., amp_u=1e-2, amp_u_tang=None,
                 taper_radius=0.05, taper_power=2, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        V   = self.W1
        dx  = self.dx

        if amp_u_tang is None:
            amp_u_tang = 0.25 * amp_u

        def _smooth_random(V, ell, amp):
            ur = TrialFunction(V); vr = TestFunction(V)
            fr = Function(V)
            fr.vector().set_local(2.0*rng.rand(V.dim()) - 1.0)
            fr.vector().apply("insert")
            a = (ur*vr + (ell**2)*dot(grad(ur), grad(vr))) * dx
            L = amp * fr * vr * dx
            uh = Function(V)
            solve(a == L, uh, solver_parameters={"linear_solver":"cg","preconditioner":"hypre_amg"})
            return uh

        ux = _smooth_random(V, ell_u, amp_u_tang)
        uy = _smooth_random(V, ell_u, amp_u_tang)
        uz = _smooth_random(V, ell_u, amp_u)

        u_vec  = as_vector((ux, uy, uz))
        Uspace = self.Q.sub(0).collapse()
        u_enr  = project(u_vec, Uspace)

        # apply taper (if requested) on u as a vector field
        if taper_radius is not None:
            u_enr = self.taper_vector(u_enr, radius=taper_radius, power=taper_power, out_space=Uspace)

        # clamp on u-subspace
        bc_u = DirichletBC(Uspace, Constant((0.0, 0.0, 0.0)), self.perim_cb)
        with silence_everything():
            bc_u.apply(u_enr.vector())

        return u_enr


    def external_force(self, P):
        """
        Return the body load vector field f(x) used in Wext = ∫ (1-alpha) f·u dx_h
        as a UFL vector expression (dim 3).
        """
        R = sqrt(self.X[0]**2 + self.X[1]**2)
        fz = (-P*self.rho_g*1e-4) \
             + (-P/5.0) * self.mg_org_per_column * exp(-(R**4)/(0.06**4))
        f_vec = as_vector((0.0, 0.0, fz))
        return (1.0 - self.alpha) * f_vec  # matches Wext exactly

    def external_force_function(self, P):
        f_expr = self.external_force(P)
        return project(f_expr, self.V_phi)   # Vector CG2 (3 components)

    def set_reference(self, q_ref, project_to_Q=True):
        """
        Store a reference mixed Function q_ref in self.Q for regularization.
        If project_to_Q=True and q_ref is not on self.Q, project its blocks to
        canonical subspaces and assemble with a persistent assigner.
        """
        # Fast path: already on the same mixed space
        if isinstance(q_ref, Function) and q_ref.function_space().id() == self.Q.id():
            self.q_ref = q_ref.copy(deepcopy=True)
            return

        if not project_to_Q:
            raise ValueError("q_ref must be a Function(self.Q) or set project_to_Q=True.")

        # Ensure canonical spaces/assigners exist
        self._ensure_canonical_spaces()

        # Split and project blocks to canonical subspaces
        uref_expr, bref_expr = split(q_ref) if not isinstance(q_ref, Function) else split(q_ref)
        u_proj = project(uref_expr, self.Uspace)
        b_proj = project(bref_expr, self.Bspace)

        # Assemble onto the mixed space without manual DOF slicing
        q_new = Function(self.Q)
        self.assign_q_from_parts.assign(q_new, [u_proj, b_proj])
        self.q_ref = q_new


    # Convenience: set reference from a displacement-only input
    def set_reference_from_u(self, u_ref,
                             beta_init=None,
                             newton_max_it=25, newton_tol=1e-10,
                             project_to_Q=True):
        q_ref, _, _ = self._assemble_q_from_u(
            u_ref, beta_init=beta_init,
            newton_max_it=newton_max_it, newton_tol=newton_tol
        )
        self.set_reference(q_ref, project_to_Q=False)

    def balanced_residual(self, q_, P, comb_alpha = 0.5,
                      reg_kind="H1",
                      u_weight=1.0, beta_weight=1.0, surface_weight=True,
                      q_ref=None, eps=1e-16,
                      return_function=True):
        """
        Build a 'balanced' gradient direction by renormalizing the post-metric
        components to match their pre-metric norms:
          r_bal = s_el * r_el + s_reg * r_reg,
          with s_* = ||b_*|| / ||r_*||.
        Returns: (r_bal Function) if return_function else (b_bal PETScVector).
        """
        # Forms
        Pi_el = self._elastic_energy(q_)
        R_el  = derivative(Pi_el, q_, TestFunction(self.Q))
        b_el  = assemble(R_el);  
        with silence_everything():
            [bc.apply(b_el) for bc in self.bcs]

        Reg_form = self.regularizer(q_, q_ref=q_ref, kind=reg_kind,
                                    surface_weight=surface_weight,
                                    u_weight=u_weight, beta_weight=beta_weight)
        R_reg = derivative(Reg_form, q_, TestFunction(self.Q))
        b_reg = assemble(R_reg); 
        with silence_everything():
            [bc.apply(b_reg) for bc in self.bcs]

        # Post-metric gradients
        r_el  = Function(self.Q); self.M_solver.solve(r_el.vector(),  b_el)
        r_reg = Function(self.Q); self.M_solver.solve(r_reg.vector(), b_reg)

        vn = 'l2'
        nb_el  = b_el.norm(vn); nr_el  = r_el.vector().norm(vn)
        nb_reg = b_reg.norm(vn); nr_reg = r_reg.vector().norm(vn)

        s_el  = (1-comb_alpha) * nb_el  / (nr_el  + eps)
        s_reg = comb_alpha * nb_reg / (nr_reg + eps)

        if return_function:
            r_bal = Function(self.Q)
            r_bal.vector().zero(); r_bal.vector().axpy(s_el,  r_el.vector())
            r_bal.vector().axpy(s_reg, r_reg.vector())
            r_bal.vector().apply("insert")
            return r_bal
        else:
            b_bal = b_el.copy()
            b_bal *= 0.0
            b_bal.axpy(s_el,  b_el)
            b_bal.axpy(s_reg, b_reg)
            return b_bal

    # -- helper: get u_ref in the enriched displacement subspace -----------
    def _u_ref_in_U(self):
        """
        Return u_ref on the canonical Uspace (builds canonical spaces if needed).
        """
        self._ensure_canonical_spaces()
        try:
            uref_expr, _ = split(self.q_ref)
            return project(uref_expr, self.Uspace)
        except Exception:
            u0 = Function(self.Uspace)
            u0.vector().zero(); u0.vector().apply("insert")
            return u0

    def _helmholtz_filter_u(self, r_u, ell_smooth):
        """
        Solve (I - ell^2 Δ) r_s = r_u with homogeneous Dirichlet on u's boundary.
        Returns a smoothed vector field in the same space as r_u.
        """
        with silence_everything():
            Uspace = r_u.function_space()
            rs = TrialFunction(Uspace); v = TestFunction(Uspace)
            a = (inner(rs, v) + (ell_smooth**2) * inner(grad(rs), grad(v))) * self.dx
            L = inner(r_u, v) * self.dx
            rs_fun = Function(Uspace)
            bc_u = DirichletBC(Uspace, Constant((0.0, 0.0, 0.0)), self.perim_cb)
            solve(a == L, rs_fun, bc_u,
                  solver_parameters={"linear_solver":"cg","preconditioner":"hypre_amg"})
            return rs_fun


    def regularizer(self, q_, q_ref=None, kind="L2", surface_weight=True,
                    u_weight=1.0, beta_weight=1.0):
        """
        Build a UFL Form R(q;q_ref) suitable for derivative().
        kind: "L2" or "H1"
        surface_weight: include sqrt(j0) in the integrand if True.
        """
        if q_ref is None:
            if not hasattr(self, "q_ref"):
                raise ValueError("No reference set. Call set_reference(q_ref) or pass q_ref=...")
            q_ref = self.q_ref

        u,  b  = split(q_)
        ur, br = split(q_ref)

        du = u - ur
        db = b - br

        weight = sqrt(self.j0) if surface_weight else 1.0

        if kind.lower() == "l2":
            integrand = 0.5 * (u_weight*inner(du, du) + beta_weight*inner(db, db))
            return (integrand * weight) * self.dx

        if kind.lower() == "h1":
            Lchar = Constant(self.mesh.hmax()**2)   # units length^2
            kappa_u = 1.0
            integrand = 0.5 * (
                u_weight*inner(du, du) + beta_weight*inner(db, db)
              + kappa_u*(u_weight*inner(grad(du), grad(du)) + beta_weight*inner(grad(db), grad(db)))
            )
            return (integrand * weight) * self.dx

        raise ValueError("kind must be 'L2' or 'H1'")

    def _get_scalar_helmholtz_solver(self, mesh, ell_smooth, Vscalar, bc_scalar,
                                     rtol=1e-5, max_it=300, cache=True):
        """
        Build (and optionally cache) a scalar Helmholtz solver on 'mesh' for space Vscalar:
            (I - ell^2 Δ) x = rhs, Dirichlet(bc_scalar)
        Returns: (A, solver)
        """
        # safer cache key
        key = (mesh.id(), id(Vscalar), float(ell_smooth))
        if cache and hasattr(self, "_helmholtz_cache") and key in self._helmholtz_cache:
            return self._helmholtz_cache[key]

        dx_loc = Measure("dx", domain=mesh)
        u = TrialFunction(Vscalar)
        v = TestFunction(Vscalar)
        A_form = (u*v + (ell_smooth**2) * dot(grad(u), grad(v))) * dx_loc
        A = assemble(A_form)
        bc_scalar.apply(A)

        solver = PETScKrylovSolver("cg", "hypre_amg")
        solver.set_operator(A)

        # --- version-compatible tolerances ---
        try:
            # newer dolfin (if available)
            solver.set_tolerances(rtol=rtol, atol=0.0, max_it=max_it)  # may raise AttributeError
        except AttributeError:
            try:
                # classic dolfin parameter dict
                solver.parameters["relative_tolerance"] = rtol
                solver.parameters["absolute_tolerance"] = 0.0
                solver.parameters["maximum_iterations"] = max_it
                solver.parameters["monitor_convergence"] = False
            except Exception:
                # fallback via PETSc KSP (petsc4py)
                ksp = solver.ksp()
                # setTolerances(rtol=None, atol=None, divtol=None, max_it=None)
                ksp.setTolerances(rtol=rtol, atol=0.0, max_it=max_it)

        if cache:
            if not hasattr(self, "_helmholtz_cache"):
                self._helmholtz_cache = {}
            self._helmholtz_cache[key] = (A, solver)

        return A, solver

    def _solve_scalar_helmholtz(self, mesh, ell_smooth, rhs_fun, rtol=1e-5, max_it=300, cache=True):
        """
        Solve (I - ell^2 Δ) x = rhs_fun with Dirichlet 0 using a cached scalar solver.
        rhs_fun must be a scalar Function/expr on mesh (we assemble rhs = ∫ rhs_fun * v dx).
        """
        Vscalar = FunctionSpace(mesh, "CG", 1)
        v = TestFunction(Vscalar)
        dx_loc = Measure("dx", domain=mesh)

        # Dirichlet BC in this scalar space
        bc_scalar = DirichletBC(Vscalar, Constant(0.0), self.perim_cb)

        # RHS: ∫ rhs_fun * v dx   (don't use inner(...) for scalars)
        if isinstance(rhs_fun, Function) and rhs_fun.function_space().mesh().id() != mesh.id():
            # make sure it's on this mesh
            rhs_fun = project(rhs_fun, Vscalar)
        b = assemble(rhs_fun * v * dx_loc)
        bc_scalar.apply(b)

        # Get (or build) solver
        A, solver = self._get_scalar_helmholtz_solver(mesh, ell_smooth, Vscalar, bc_scalar,
                                                      rtol=rtol, max_it=max_it, cache=cache)

        x = Function(Vscalar)
        solver.solve(x.vector(), b)
        return x


    def helmholtz_filter_u_fast(self, r_u, ell_smooth,
                                rtol=1e-5, max_it=300, cache=True,
                                work_space="cg1",  # "same" or "cg1"
                                project_back=True):
        """
        Fast Helmholtz smoothing of a vector field:
          (I - ell^2 Δ) r_s = r_u,  Dirichlet 0 on clamped boundary.

        work_space:
          - "cg1": project r_u to Vector CG1, solve 3 scalar problems with a cached AMG, then (optionally) project back.
          - "same": solve in r_u's original vector space (slower).

        Returns: smoothed vector Function in the SAME space as r_u if project_back=True,
                 else in the chosen work space.
        """
        with silence_everything():
            # Choose mesh & original space
            if not isinstance(r_u, Function):
                raise ValueError("helmholtz_filter_u_fast expects r_u as a Function.")
            mesh = r_u.function_space().mesh()

            if work_space == "same":
                # Original method with looser tolerances + AMG (simple path)
                U = r_u.function_space()
                rs = TrialFunction(U); v = TestFunction(U)
                dx_loc = Measure("dx", domain=mesh)
                a = (inner(rs, v) + (ell_smooth**2) * inner(grad(rs), grad(v))) * dx_loc
                L = inner(r_u, v) * dx_loc
                out = Function(U)
                bc_u = DirichletBC(U, Constant((0.0, 0.0, 0.0)), self.perim_cb)
                solve(a == L, out, bc_u, solver_parameters={
                    "linear_solver": "cg",
                    "preconditioner": "hypre_amg",
                    "convergence_criterion": "residual",
                    "relative_tolerance": rtol,
                    "absolute_tolerance": 0.0,
                    "maximum_iterations": max_it,
                })
                return out

            # work_space == "cg1": cheap vector space
            Vvec = VectorFunctionSpace(mesh, "CG", 1, dim=3)
            r_u_cg1 = project(r_u, Vvec)

            # Split into components and solve scalar Helmholtz for each (reuse AMG)
            r0, r1, r2 = r_u_cg1.split(deepcopy=True)
            s0 = self._solve_scalar_helmholtz(mesh, ell_smooth, r0, rtol=rtol, max_it=max_it, cache=cache)
            s1 = self._solve_scalar_helmholtz(mesh, ell_smooth, r1, rtol=rtol, max_it=max_it, cache=cache)
            s2 = self._solve_scalar_helmholtz(mesh, ell_smooth, r2, rtol=rtol, max_it=max_it, cache=cache)

            s_vec = as_vector((s0, s1, s2))
            s_vec_fun = project(s_vec, Vvec)

            if project_back and (r_u.function_space().id() != Vvec.id()):
                return project(s_vec_fun, r_u.function_space())
            else:
                return s_vec_fun

    def _ensure_canonical_spaces(self):
        """
        Create (once) canonical collapsed subspaces and persistent assigners so we
        never rely on ad-hoc collapsed spaces (which can reorder DOFs).
        """
        if hasattr(self, "Uspace") and hasattr(self, "Bspace"):
            return
        # Canonical collapsed subspaces (build once, reuse everywhere)
        self.Uspace = self.Q.sub(0).collapse()
        self.Bspace = self.Q.sub(1).collapse()
        # Persistent assigners
        self.assign_q_from_parts = FunctionAssigner(self.Q, [self.Uspace, self.Bspace])

    def plot(self, y, n=None, title=None, fname='temp.png'):
        """
        y : Vector Function *or* UFL vector expression for the mid-surface (x,y,z).
        n : (optional) Vector Function *or* UFL vector expression for normals.
        Returns: matplotlib Axes3D
        """
        with silence_everything():
            y = y + self.phi0
            # Ensure y is a Function in the vector space we use for geometry
            if not isinstance(y, Function):
                y = project(y, self.V_phi)

            # If normals are provided as UFL, project them too
            if (n is not None) and (not isinstance(n, Function)):
                n = project(n, self.V_normal)

            y_0, y_1, y_2 = y.split(deepcopy=True)

        fig = plt.figure(figsize=[4,4])
        ax = fig.add_subplot(projection='3d')

        ax.plot_trisurf(y_0.compute_vertex_values(),
                        y_1.compute_vertex_values(),
                        y_2.compute_vertex_values(),
                        triangles=y.function_space().mesh().cells(),
                        linewidth=1, antialiased=True, shade=False)

        ax.set_axis_off()  
        ax.view_init(elev=20, azim=80)
        plt.xlabel(r"$x_0$")
        plt.ylabel(r"$x_1$")
        if title is not None:
            ax.set_title(title, y=0.9, fontdict = {'weight':'bold'}) 
        plt.tight_layout()
        plt.savefig(fname,bbox_inches='tight', pad_inches=0.02)
        plt.close()


if __name__ == '__main__':
    # 1) Build the energy object (point it to your mesh file)
    energy = ShellEnergy(xdmf_path="data/only_spiral.xdmf", use_spiral_perim=True)

    # visualize baseline geometry (mid-surface y = u + phi0)
    #u_prox, _ = split(energy.q_ref)
    u_prox = energy.q_ref.sub(0, deepcopy=True)
    energy.plot(u_prox, title='Baseline (u only)', fname = 'u_prox.png')

    # 2) Sample a random displacement u and evaluate; evaluate solves beta* internally
    u_rand = energy.sample_m(seed=0)
    fM, g = energy.evaluate(u_rand)

    # Plot current mid-surface from u only
    energy.plot(u_rand, title=r"", fname = 'kiri_m.pdf')

    # Plot gradient's displacement block for visualization
    energy.plot(g, title=r"", fname = "kiri_g.pdf")

    unorm = u_rand.vector().norm('l2')
    gnorm = g.vector().norm('l2')
    g_small = project(Constant(unorm/gnorm)*g, g.function_space())
    #g_small.vector().norm('l2')
    energy.plot(g_small, title='grad (u-block)', fname = "grad_small.png")

    #energy.evaluate(u_prox)

    # Plot gradient's displacement block for visualization
    plot_p(energy, g, title='grad (u-block)', fname = "grad.png")

    unorm = u_rand.vector().norm('l2')
    gnorm = g.vector().norm('l2')
    g_small = project(Constant(unorm/gnorm)*g, g.function_space())
    #g_small.vector().norm('l2')
    plot_p(energy, g_small, title='grad (u-block)', fname = "grad_small.png")

