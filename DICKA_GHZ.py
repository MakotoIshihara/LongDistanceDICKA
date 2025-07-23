import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
from ncpol2sdpa.nc_utils import ncdegree, get_monomials
import mosek
import chaospy
import qutip as qtp
from scipy.optimize import minimize

def state4(q, eta, etae, pd):
  """
  Defining a distributed quantum state among the legitimate parties when successful events occur
  """

  S = sqrt(1-q)*(sqrt(eta)*qtp.ket('000'))
  V = sqrt(q)*(sqrt(etae)*qtp.ket('100') + sqrt(1-etae)*qtp.ket('001')) + sqrt(1-q)*sqrt(1-eta)*qtp.ket('010')

  Phi1 = 0.5 * (qtp.tensor(S, V, S, V) - qtp.tensor(V, S, V, S))
  Phi2 = 0.5 * (qtp.tensor(S, V, V, V) + qtp.tensor(V, S, V, V) + qtp.tensor(V, V, S, V) + qtp.tensor(V, V, V, S))
  Phi3 = 0.5 * (qtp.tensor(S, V, V, V) - qtp.tensor(V, S, V, V) + qtp.tensor(V, V, S, V) - qtp.tensor(V, V, V, S))
  Phi4 = qtp.tensor(V, V, V, V)

  Rho1 = (Phi1.proj()).ptrace([0,3,6,9])
  Rho2 = (Phi2.proj()).ptrace([0,3,6,9])
  Rho3 = (Phi3.proj()).ptrace([0,3,6,9])
  Rho4 = (Phi4.proj()).ptrace([0,3,6,9])

  P1 = ((1-pd)**2*(Phi1.proj())).tr()
  P2 = (pd*(1-pd)**2*(Phi2.proj())).tr()
  P3 = (pd*(1-pd)**2*(Phi3.proj())).tr()
  P4 = (pd**2*(1-pd)**2*(Phi4.proj())).tr()

  Rhofinal = qtp.Qobj((1/(P1+P2+P3+P4)*((1-pd)**2*Rho1 + pd*(1-pd)**2*Rho2 + pd*(1-pd)**2*Rho3 + pd**2*(1-pd)**2*Rho4)).full())

  return Rhofinal

def objective(ti):
    """"
    Defining the objective function in Eq. 11
    """
    return (A[0][0]*(Dagger(Z[0]) + Z[0]) + (1-A[0][0])*(Dagger(Z[1]) + Z[1])) + \
		(1-ti) * (A[0][0]*Dagger(Z[0])*Z[0] + (1-A[0][0])*Dagger(Z[1])*Z[1]) + \
		ti * (Z[0]*Dagger(Z[0]) + Z[1]*Dagger(Z[1]))

def score_constraints(sys, q, eta, etae, pd):
    """
    Returns moment equality constraints.
        sys --  parameterization [a0, a1, b0, b1, c0, c1, d0, d1]
        q  --  parameter of single-photon entanglement
        eta -- channel loss
        etae -- detection efficiency
        pd -- dark count probability
    """

    [id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
    [a0, a1, b0, b1, c0, c1, d0, d1] = sys[:]
    a0 = 0

    # quantum state
    rho = state4(q, eta, etae, pd)

    # Measurements of Alice and Bob
    coh10 = np.exp(-abs(a0)**2/2)*(qtp.ket('0')+a0*qtp.ket('1'))
    coh11 = np.exp(-abs(a1)**2/2)*(qtp.ket('0')+a1*qtp.ket('1'))
    coh20 = np.exp(-abs(b0)**2/2)*(qtp.ket('0')+b0*qtp.ket('1'))
    coh21 = np.exp(-abs(b1)**2/2)*(qtp.ket('0')+b1*qtp.ket('1'))
    coh30 = np.exp(-abs(c0)**2/2)*(qtp.ket('0')+c0*qtp.ket('1'))
    coh31 = np.exp(-abs(c1)**2/2)*(qtp.ket('0')+c1*qtp.ket('1'))
    coh40 = np.exp(-abs(d0)**2/2)*(qtp.ket('0')+d0*qtp.ket('1'))
    coh41 = np.exp(-abs(d1)**2/2)*(qtp.ket('0')+d1*qtp.ket('1'))

    a00 = (1-pd)*coh10.proj()
    a01 = id - a00
    a10 = (1-pd)*coh11.proj()
    a11 = id - a10
    b00 = (1-pd)*coh20.proj()
    b01 = id - b00
    b10 = (1-pd)*coh21.proj()
    b11 = id - b10
    c00 = (1-pd)*coh30.proj()
    c01 = id - c00
    c10 = (1-pd)*coh31.proj()
    c11 = id - c10
    d00 = (1-pd)*coh40.proj()
    d01 = id - d00
    d10 = (1-pd)*coh41.proj()
    d11 = id - d10

    A_meas = [[a00, a01], [a10, a11]]
    B_meas = [[b00, b01], [b10, b11]]
    C_meas = [[c00, c01], [c10, c11]]
    D_meas = [[d00, d01], [d10, d11]]

    # Constraints
    constraints = []
    for x in range(2):
        for y in range(2):
            for z in range(2):
                for w in range(2):
                    constraints += [A[x][0]*B[y][0]*C[z][0]*D[w][0] - (rho*qtp.Qobj((qtp.tensor(A_meas[x][0], B_meas[y][0], C_meas[z][0], D_meas[w][0])).full())).tr().real]

    for x in range(2):
        for y in range(2):
            for z in range(2):
                constraints += [A[x][0]*B[y][0]*C[z][0] - (rho*qtp.Qobj((qtp.tensor(A_meas[x][0], B_meas[y][0], C_meas[z][0], id)).full())).tr().real]
    for x in range(2):
        for y in range(2):
            for w in range(2):
                constraints += [A[x][0]*B[y][0]*D[w][0] - (rho*qtp.Qobj((qtp.tensor(A_meas[x][0], B_meas[y][0], id, D_meas[w][0])).full())).tr().real]
    for x in range(2):
        for z in range(2):
            for w in range(2):
                constraints += [A[x][0]*C[z][0]*D[w][0] - (rho*qtp.Qobj((qtp.tensor(A_meas[x][0], id, C_meas[z][0], D_meas[w][0])).full())).tr().real]
    for y in range(2):
        for z in range(2):
            for w in range(2):
                constraints += [B[y][0]*C[z][0]*D[w][0] - (rho*qtp.Qobj((qtp.tensor(id, B_meas[y][0], C_meas[z][0], D_meas[w][0])).full())).tr().real]

    for x in range(2):
        for y in range(2):
                constraints += [A[x][0]*B[y][0] - (rho*qtp.Qobj((qtp.tensor(A_meas[x][0], B_meas[y][0], id, id)).full())).tr().real]
    for x in range(2):
        for z in range(2):
                constraints += [A[x][0]*C[z][0] - (rho*qtp.Qobj((qtp.tensor(A_meas[x][0], id, C_meas[z][0], id)).full())).tr().real]
    for x in range(2):
        for w in range(2):
                constraints += [A[x][0]*D[w][0] - (rho*qtp.Qobj((qtp.tensor(A_meas[x][0], id, id, D_meas[w][0])).full())).tr().real]
    for y in range(2):
        for z in range(2):
                constraints += [B[y][0]*C[z][0] - (rho*qtp.Qobj((qtp.tensor(id, B_meas[y][0], C_meas[z][0], id)).full())).tr().real]
    for y in range(2):
        for w in range(2):
                constraints += [B[y][0]*D[w][0] - (rho*qtp.Qobj((qtp.tensor(id, B_meas[y][0], id, D_meas[w][0])).full())).tr().real]
    for z in range(2):
        for w in range(2):
                constraints += [C[z][0]*D[w][0] - (rho*qtp.Qobj((qtp.tensor(id, id, C_meas[z][0], D_meas[w][0])).full())).tr().real]

    constraints += [A[0][0] - (rho*qtp.Qobj((qtp.tensor(A_meas[0][0], id, id, id)).full())).tr().real]
    constraints += [B[0][0] - (rho*qtp.Qobj((qtp.tensor(id, B_meas[0][0], id, id)).full())).tr().real]
    constraints += [C[0][0] - (rho*qtp.Qobj((qtp.tensor(id, id, C_meas[0][0], id)).full())).tr().real]
    constraints += [D[0][0] - (rho*qtp.Qobj((qtp.tensor(id, id, id, D_meas[0][0])).full())).tr().real]
    constraints += [A[1][0] - (rho*qtp.Qobj((qtp.tensor(A_meas[1][0], id, id, id)).full())).tr().real]
    constraints += [B[1][0] - (rho*qtp.Qobj((qtp.tensor(id, B_meas[1][0], id, id)).full())).tr().real]
    constraints += [C[1][0] - (rho*qtp.Qobj((qtp.tensor(id, id, C_meas[1][0], id)).full())).tr().real]
    constraints += [D[1][0] - (rho*qtp.Qobj((qtp.tensor(id, id, id, D_meas[1][0])).full())).tr().real]

    return constraints[:]

def sys2vec(sys, q, eta, etae, pd):
    """
    Returns vector of probabilities p(0000|xyzw) associated with sys in the same order
    as the constraints are specified in score_constraints function.
    """
    [id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
    [a0, a1, b0, b1, c0, c1, d0, d1] = sys[:]
    a0 = 0
    
    # quantum state
    rho = state4(q, eta, etae, pd)

    # Measurements of Alice and Bob
    coh10 = np.exp(-abs(a0)**2/2)*(qtp.ket('0')+a0/sqrt(etae)*qtp.ket('1'))
    coh11 = np.exp(-abs(a1)**2/2)*(qtp.ket('0')+a1/sqrt(etae)*qtp.ket('1'))
    coh20 = np.exp(-abs(b0)**2/2)*(qtp.ket('0')+b0/sqrt(etae)*qtp.ket('1'))
    coh21 = np.exp(-abs(b1)**2/2)*(qtp.ket('0')+b1/sqrt(etae)*qtp.ket('1'))
    coh30 = np.exp(-abs(c0)**2/2)*(qtp.ket('0')+c0/sqrt(etae)*qtp.ket('1'))
    coh31 = np.exp(-abs(c1)**2/2)*(qtp.ket('0')+c1/sqrt(etae)*qtp.ket('1'))
    coh40 = np.exp(-abs(d0)**2/2)*(qtp.ket('0')+d0/sqrt(etae)*qtp.ket('1'))
    coh41 = np.exp(-abs(d1)**2/2)*(qtp.ket('0')+d1/sqrt(etae)*qtp.ket('1'))

    a00 = (1-pd)*coh10.proj()
    a01 = id - a00
    a10 = (1-pd)*coh11.proj()
    a11 = id - a10
    b00 = (1-pd)*coh20.proj()
    b01 = id - b00
    b10 = (1-pd)*coh21.proj()
    b11 = id - b10
    c00 = (1-pd)*coh30.proj()
    c01 = id - c00
    c10 = (1-pd)*coh31.proj()
    c11 = id - c10
    d00 = (1-pd)*coh40.proj()
    d01 = id - d00
    d10 = (1-pd)*coh41.proj()
    d11 = id - d10

    A_meas = [[a00, a01], [a10, a11]]
    B_meas = [[b00, b01], [b10, b11]]
    C_meas = [[c00, c01], [c10, c11]]
    D_meas = [[d00, d01], [d10, d11]]

    vec = []
    for x in range(2):
        for y in range(2):
            for z in range(2):
                for w in range(2):
                    vec += [(rho*qtp.Qobj((qtp.tensor(A_meas[x][0], B_meas[y][0], C_meas[z][0], D_meas[w][0])).full())).tr().real]

    # Marginal constraints
    for x in range(2):
        for y in range(2):
            for z in range(2):
                vec += [(rho*qtp.Qobj((qtp.tensor(A_meas[x][0], B_meas[y][0], C_meas[z][0], id)).full())).tr().real]
    for x in range(2):
        for y in range(2):
            for w in range(2):
                vec += [(rho*qtp.Qobj((qtp.tensor(A_meas[x][0], B_meas[y][0], id, D_meas[w][0])).full())).tr().real]
    for x in range(2):
        for z in range(2):
            for w in range(2):
                vec += [(rho*qtp.Qobj((qtp.tensor(A_meas[x][0], id, C_meas[z][0], D_meas[w][0])).full())).tr().real]
    for y in range(2):
        for z in range(2):
            for w in range(2):
                vec += [(rho*qtp.Qobj((qtp.tensor(id, B_meas[y][0], C_meas[z][0], D_meas[w][0])).full())).tr().real]

    for x in range(2):
        for y in range(2):
                vec += [(rho*qtp.Qobj((qtp.tensor(A_meas[x][0], B_meas[y][0], id, id)).full())).tr().real]
    for x in range(2):
        for z in range(2):
                vec += [(rho*qtp.Qobj((qtp.tensor(A_meas[x][0], id, C_meas[z][0], id)).full())).tr().real]
    for x in range(2):
        for w in range(2):
                vec += [(rho*qtp.Qobj((qtp.tensor(A_meas[x][0], id, id, D_meas[w][0])).full())).tr().real]
    for y in range(2):
        for z in range(2):
                vec += [(rho*qtp.Qobj((qtp.tensor(id, B_meas[y][0], C_meas[z][0], id)).full())).tr().real]
    for y in range(2):
        for w in range(2):
                vec += [(rho*qtp.Qobj((qtp.tensor(id, B_meas[y][0], id, D_meas[w][0])).full())).tr().real]
    for z in range(2):
        for w in range(2):
                vec += [(rho*qtp.Qobj((qtp.tensor(id, id, C_meas[z][0], D_meas[w][0])).full())).tr().real]

    vec += [(rho*qtp.Qobj((qtp.tensor(A_meas[0][0], id, id, id)).full())).tr().real]
    vec += [(rho*qtp.Qobj((qtp.tensor(id, B_meas[0][0], id, id)).full())).tr().real]
    vec += [(rho*qtp.Qobj((qtp.tensor(id, id, C_meas[0][0], id)).full())).tr().real]
    vec += [(rho*qtp.Qobj((qtp.tensor(id, id, id, D_meas[0][0])).full())).tr().real]
    vec += [(rho*qtp.Qobj((qtp.tensor(A_meas[1][0], id, id, id)).full())).tr().real]
    vec += [(rho*qtp.Qobj((qtp.tensor(id, B_meas[1][0], id, id)).full())).tr().real]
    vec += [(rho*qtp.Qobj((qtp.tensor(id, id, C_meas[1][0], id)).full())).tr().real]
    vec += [(rho*qtp.Qobj((qtp.tensor(id, id, id, D_meas[1][0])).full())).tr().real]

    return vec

def sdp_dual_vec(SDP):
    """
    Extracting dual vector. 80 is the number of constraints. If we change the number constraints, we should change this value.
    """
    raw_vec = SDP.y_mat[-160:]
    vec = [raw_vec[2*k][0][0] - raw_vec[2*k + 1][0][0] for k in range(80)]
    return np.array(vec[:])

def dual_vec_to_improved_sys(dvec, init_sys, q, eta, etae, pd):
    """
    Optimizing system
    """
    def f0(x):
        p = sys2vec(x, q, eta, etae, pd)
        return -np.dot(p, dvec)

    bounds = [[-5, 5]] + [[-5,5] for _ in range(len(init_sys) - 1)]
    res = minimize(f0, init_sys[:], bounds = bounds)

    return res.x.tolist()[:]

def optimise_rate(SDP, sys, q, eta, etae, pd):
    """
    Optimizing key rate
    """
    NEEDS_IMPROVING = True
    FIRST_PASS = True
    improved_sys = sys[:]
    best_sys = sys[:]
    dual_vec = np.zeros(80)

    while(NEEDS_IMPROVING):
        if not FIRST_PASS:
            improved_sys = dual_vec_to_improved_sys(dual_vec, improved_sys[:], q, eta, etae, pd)

        score_cons = score_constraints(improved_sys[:], q, eta, etae, pd)
        SDP.process_constraints(equalities = op_eqs,
                            inequalities = op_ineqs,
                            momentequalities = moment_eqs[:] + score_cons[:],
                            momentinequalities = moment_ineqs)
        dual_vec, new_ent = compute_dual_vector(SDP, improved_sys[:])

        if not FIRST_PASS:
            if new_ent < current_ent + current_ent*EPS_M or new_ent < current_ent + EPS_A :
                NEEDS_IMPROVING = False
        else:
            starting_ent = new_ent
            current_ent = new_ent
            FIRST_PASS = False

        if new_ent > current_ent:
            if VERBOSE > 0:
                print('Optimizing sys for eta =', eta, ' ... ', starting_ent, '->', new_ent)
                print(improved_sys)
            current_ent = new_ent
            best_sys = improved_sys[:]

    return current_ent, best_sys[:]

def compute_entropy(SDP):
    """
    Computing lower bound on the conditional von Neumann entropy H(A|x^*, E)
    """
    ck = 0.0
    ent = 0.0

    for k in range(len(T)-1):
        ck = W[k]/(T[k] * log(2))
        new_objective = objective(T[k])
        SDP.set_objective(new_objective)
        SDP.solve('mosek')

        if SDP.status == 'optimal':
            ent += ck * (1 + SDP.dual)
        else:
            ent = 0
            if VERBOSE:
                print('Bad solve: ', k, SDP.status)
            break

    return ent

def compute_dual_vector(SDP, sys):
    """
    Computing dual vector
    """
    dual_vec = np.zeros(80)
    ck = 0.0
    ent = 0.0

    for k in range(len(T)-1):
        ck = W[k]/(T[k] * log(2))
        new_objective = objective(T[k])
        SDP.set_objective(new_objective)
        SDP.solve('mosek')

        if SDP.status == 'optimal':
            ent += ck * (1 + SDP.dual)
            d = sdp_dual_vec(SDP)
            dual_vec = dual_vec + ck * d
        else:
            ent = 0
            dual_vec = np.zeros(80)
            break

    return dual_vec, ent


def get_subs():
    """"
    Constraints on measurement operators. Measurements of the legitimate parties are projective, and commute with Eve's measurements
    """
    subs = {}
    subs.update(ncp.projective_measurement_constrains(A,B,C,D))
    for a in ncp.flatten([A,B,C,D]):
        for z in Z:
            subs.update({z*a : a*z, Dagger(z)*z : a*Dagger(z)})
    
    return subs


def get_extra_monomials_fast():
     """
     Preparing extra monomials for the optimization
     """
     monos = []
     ZZ = Z + [Dagger(z) for z in Z]
     Aflat = ncp.flatten(A)
     Bflat = ncp.flatten(B)
     Cflat = ncp.flatten(C)
     Dflat = ncp.flatten(D)
     ABCD = [z for z in get_monomials(Aflat+Bflat+Cflat+Dflat,4) if ncdegree(z)>=4]

     monos += [A[0][0]*Dagger(Z[0])*Z[0]]
     monos += [A[0][0]*Dagger(Z[1])*Z[1]]

     return monos[:] + ABCD[:]

def generate_quadrature(m):
    """
    Generating the Gauss-Radau quadrature nodes t and weights w. Note that the number of nodes is 2*m
    """
    t, w = chaospy.quadrature.radau(m, chaospy.Uniform(0, 1), 1)
    t = t[0]
    return t, w

# number of outputs for each inputs of devices of the legitimate parties
A_config = [2,2]
B_config = [2,2]
C_config = [2,2]
D_config = [2,2]

# Defining operators
A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
C = [Ck for Ck in ncp.generate_measurements(C_config, 'C')]
D = [Dl for Dl in ncp.generate_measurements(D_config, 'D')]
Z = ncp.generate_operators('Z', 2, hermitian=0)

#Defining parameters
LEVEL = 2                        # NPA relaxation level
M = 2                            # Number of nodes / 2 in gaussian quadrature
T, W = generate_quadrature(M)    # Nodes, weights of quadrature
VERBOSE = 1                      # If > 1 then ncpol2sdpa will also be verbose
EPS_M, EPS_A = 1e-3, 1e-3        # Multiplicative/Additive epsilon in iterative optimization
eta = 0.9                        # Channel efficiency
etae = 0.99                      # Detection efficiency
pd = 1e-6                        # Dark count probability
q = 0.95                         # Parameter of single-photon entanglement

substitutions = get_subs()               # Projection & commutation relation
moment_ineqs = []                        # Moment inequalities (not needed here)
moment_eqs = []                          # Moment equalities (not needed here)
op_eqs = []                              # Operator equalities (not needed here)
op_ineqs = []                            # Operator inequalities (not needed here)
extra_monos = get_extra_monomials_fast() # Extra monomials

#Initial system
test_sys = [0.0, -0.6325855412762315, 0.2013409785300114, -0.5530833238475724, 0.4752809808226817, -0.2698989524666235, 0.49299923263621226, -0.2824816843135141]
score_cons = score_constraints(test_sys, q, eta, etae, pd)
ops = ncp.flatten([A,B,C,D,Z])
obj = objective(1)

#Constructing SDP
sdp = ncp.SdpRelaxation(ops, verbose = VERBOSE-1, normalized=True, parallel=0)
sdp.get_relaxation(level = LEVEL,
                    equalities = op_eqs[:],
                    inequalities = op_ineqs[:],
                    momentequalities = moment_eqs[:] + score_cons[:],
                    momentinequalities = moment_ineqs[:],
                    objective = obj,
                    substitutions = substitutions,
                    extramonomials = extra_monos)

#Solving SDP
opt_rate, opt_sys = optimise_rate(sdp, test_sys[:], q, eta, etae, pd)
print(opt_rate)
print(opt_sys)