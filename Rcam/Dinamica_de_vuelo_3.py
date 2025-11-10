import numpy as np
import matplotlib.pyplot as plt

def RCAM_model(X:np.ndarray, U:np.ndarray, rho:float) -> np.ndarray:

    #------------------------ constants -------------------------------

    # Nominal vehicle constants
    m = 120000; # kg - total mass
    
    cbar = 6.6 # m - mean aerodynamic chord
    lt = 24.8 # m - tail AC distance to CG
    S = 260 # m2 - wing area
    St = 64 # m2 - tail area
    
    # centre of gravity position
    Xcg = 0.23 * cbar # m - x pos of CG in Fm
    Ycg = 0.0 # m - y pos of CG in Fm
    Zcg = 0.10 * cbar # m - z pos of CG in Fm ERRATA - table 2.4 has 0.0 and table 2.5 has 0.10 cbar
    
    # aerodynamic center position
    Xac = 0.12 * cbar # m - x pos of aerodynamic center in Fm
    Yac = 0.0 # m - y pos of aerodynamic center in Fm
    Zac = 0.0 # m - z pos of aerodynamic center in Fm
    
    # engine point of thrust aplication
    Xapt1 = 0 # m - x position of engine 1 in Fm
    Yapt1 = -7.94 # m - y position of engine 1 in Fm
    Zapt1 = -1.9 # m - z position of engine 1 in Fm
    
    Xapt2 = 0 # m - x position of engine 2 in Fm
    Yapt2 = 7.94 # m - y position of engine 2 in Fm
    Zapt2 = -1.9 # m - z position of engine 2 in Fm
    
    # other constants
    #rho = 1.225 # kg/m3 - air density
    g = 9.81 # m/s2 - gravity
    depsda = 0.25 # rad/rad - change in downwash wrt alpha
    deg2rad = np.pi / 180 # from degrees to radians
    alpha_L0 = -11.5 * deg2rad # rad - zero lift AOA
    n = 5.5 # adm - slope of linear ragion of lift slope
    a3 = -768.5 # adm - coeff of alpha^3
    a2 = 609.2 # adm -  - coeff of alpha^2
    a1 = -155.2 # adm -  - coeff of alpha^1
    a0 = 15.212 # adm -  - coeff of alpha^0 ERRATA RCAM has 15.2
    alpha_switch = 14.5 * deg2rad # rad - kink point of lift slope
    
    
    #----------------- intermediate variables ---------------------------
    # airspeed
    Va = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2) # m/s
    
    # alpha and beta
    #np.arctan2 --> y, x
    alpha = np.arctan2(X[2], X[0])
    beta = np.arcsin(X[1]/Va)
    
    # dynamic pressure
    Q = 0.5 * rho * Va**2
    
    # define vectors wbe_b and V_b
    wbe_b = np.array([X[3], X[4], X[5]])
    V_b = np.array([X[0], X[1], X[2]])
    
    #----------------- aerodynamic force coefficients ---------------------
    # CL - wing + body
    CL_wb = n * (alpha - alpha_L0) if alpha <= alpha_switch else a3 * alpha**3 + a2 * alpha**2 + a1 * alpha + a0
    
    # CL thrust
    epsilon = depsda * (alpha - alpha_L0)
    alpha_t = alpha - epsilon + U[1] + 1.3 * X[4] * lt / Va
    CL_t = 3.1 * (St / S) * alpha_t
    
    # Total CL
    CL = CL_wb + CL_t
    
    # Total CD
    CD = 0.13 + 0.07 * (n * alpha + 0.654)**2
    
    # Total CY
    CY = -1.6 * beta + 0.24 * U[2]
    
    
    #------------------- dimensional aerodynamic forces --------------------
    # forces in F_s
    FA_s = np.array([-CD * Q* S, CY * Q * S, -CL * Q * S])
    
    # rotate forces to body axis (F_b)
    C_bs = np.array([[np.cos(alpha), 0.0, -np.sin(alpha)],
                     [0.0, 1.0, 0.0],
                     [np.sin(alpha), 0.0, np.cos(alpha)]], dtype=np.dtype('f8'))

    FA_b = np.dot(C_bs, FA_s)   
    
    
    #------------------ aerodynamic moment coefficients about AC -----------
    # moments in F_b
    eta11 = -1.4 * beta
    eta21 = -0.59 - (3.1 * (St * lt) / (S * cbar)) * (alpha - epsilon)
    eta31 = (1 - alpha * (180 / (15 * np.pi))) * beta
    
    eta = np.array([eta11, eta21, eta31])
    
    dCMdx = (cbar / Va) * np.array([[-11.0, 0.0, 5.0], 
                                    [ 0.0, (-4.03 * (St * lt**2) / (S * cbar**2)), 0.0], 
                                    [1.7, 0.0, -11.5]], dtype=np.dtype('f8'))
    dCMdu = np.array([[-0.6, 0.0, 0.22],
                      [0.0, (-3.1 * (St * lt) / (S * cbar)), 0.0],
                      [0.0, 0.0, -0.63]], dtype=np.dtype('f8'))
    
    
    # CM about AC in Fb
    CMac_b = eta + np.dot(dCMdx, wbe_b) + np.dot(dCMdu, np.array([U[0], U[1], U[2]]))
    
    #------------------- aerodynamic moment about AC -------------------------
    # normalize to aerodynamic moment
    MAac_b = CMac_b * Q * S * cbar
    
    #-------------------- aerodynamic moment about CG ------------------------
    rcg_b = np.array([Xcg, Ycg, Zcg])
    rac_b = np.array([Xac, Yac, Zac])
    
    MAcg_b = MAac_b + np.cross(FA_b, rcg_b - rac_b)
    
    #---------------------- engine force and moment --------------------------
    # thrust
    F1 = U[3] * m * g
    F2 = U[4] * m * g
    
    #thrust vectors (assuming aligned with x axis)
    FE1_b = np.array([F1, 0, 0])
    FE2_b = np.array([F2, 0, 0])
    
    FE_b = FE1_b + FE2_b
    
    # engine moments
    mew1 = np.array([Xcg - Xapt1, Yapt1 - Ycg, Zcg - Zapt1])
    mew2 = np.array([Xcg - Xapt2, Yapt2 - Ycg, Zcg - Zapt2])
    
    MEcg1_b = np.cross(mew1, FE1_b)
    MEcg2_b = np.cross(mew2, FE2_b)
    
    MEcg_b = MEcg1_b + MEcg2_b
    
    #---------------------- gravity effects ----------------------------------
    g_b = np.array([-g * np.sin(X[7]), g * np.cos(X[7]) * np.sin(X[6]), g * np.cos(X[7]) * np.cos(X[6])])
    
    Fg_b = m * g_b
    
    #---------------------- state derivatives --------------------------------
    # inertia tensor
    Ib = m * np.array([[40.07, 0.0, -2.0923],
                       [0.0, 64.0, 0.0],  
                       [-2.0923, 0.0, 99.92]], dtype=np.dtype('f8')) # ERRATA on Ixz p. 12 vs p. 91
    invIb = np.linalg.inv(Ib)
    
    # form F_b and calculate u, v, w dot
    F_b = Fg_b + FE_b + FA_b
    
    x0x1x2_dot  = (1 / m) * F_b - np.cross(wbe_b, V_b)
    
    # form Mcg_b and calc p, q r dot
    Mcg_b = MAcg_b + MEcg_b
    
    x3x4x5_dot = np.dot(invIb, (Mcg_b - np.cross(wbe_b, np.dot(Ib , wbe_b))))
    
    #phi, theta, psi dot
    H_phi = np.array([[1.0, np.sin(X[6]) * np.tan(X[7]), np.cos(X[6]) * np.tan(X[7])],
                      [0.0, np.cos(X[6]), -np.sin(X[6])],
                      [0.0, np.sin(X[6]) / np.cos(X[7]), np.cos(X[6]) / np.cos(X[7])]], dtype=np.dtype('f8'))
    
    x6x7x8_dot = np.dot(H_phi, wbe_b)
    
    #--------------------- place in first order form --------------------------
    X_dot = np.concatenate((x0x1x2_dot, x3x4x5_dot, x6x7x8_dot))
    
    return X_dot

# -------------------------------------------------------------------------
# INTEGRACIÓN
# -------------------------------------------------------------------------
dt = 1e-3
t_final = 180
N = int(t_final / dt)

# Estado inicial (solo 9 estados)
X = np.array([85, 0, 0, 0, 0, 0, 0, 0.1, 0])  # [u,v,w,p,q,r,phi,theta,psi]

# Controles (def_A, def_E, def_R, th1, th2)
U = np.array([0.0, -0.1, 0.0, 0.08, 0.08])

# Densidad del aire
rho = 1.225

# Historial
X_hist = np.zeros((N+1, len(X)))

time = np.linspace(0, t_final, N+1)
X_hist[0, :] = X


# Integración Euler
for k in range(N):

    if 30000 <= k < 32000:  # entre 30s y 32s
        U[0] = np.deg2rad(5)  # def_A
    else:
        U[0] = 0.0

    X_dot = RCAM_model(X, U, rho)
    X = X + X_dot * dt
    X_hist[k+1, :] = X

# -------------------------------------------------------------------------
# GRAFICAR RESULTADOS
# -------------------------------------------------------------------------
plt.figure(figsize=(10,6))
plt.subplot(2,2,1); plt.plot(time, X_hist[:,0]); plt.ylabel('u [m/s]'); plt.grid(True)
plt.subplot(2,2,2); plt.plot(time, X_hist[:,2]); plt.ylabel('w [m/s]'); plt.grid(True)
plt.subplot(2,2,3); plt.plot(time, X_hist[:,4]); plt.ylabel('q [rad/s]'); plt.grid(True)
plt.subplot(2,2,4); plt.plot(time, X_hist[:,7]); plt.ylabel('theta [rad]'); plt.grid(True)
plt.suptitle('Longitudinal motion for Aileron deflection'); plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.subplot(3,2,1); plt.plot(time, X_hist[:,1]); plt.ylabel('v [m/s]'); plt.grid(True)
plt.subplot(3,2,2); plt.plot(time, X_hist[:,3]); plt.ylabel('p [rad/s]'); plt.grid(True)
plt.subplot(3,2,3); plt.plot(time, X_hist[:,5]); plt.ylabel('r [rad/s]'); plt.grid(True)
plt.subplot(3,2,4); plt.plot(time, X_hist[:,6]); plt.ylabel('phi [rad]'); plt.grid(True)
plt.subplot(3,2,5); plt.plot(time, X_hist[:,8]); plt.ylabel('psi [rad]'); plt.grid(True)
plt.suptitle('Lateral-directional motion for Aileron deflection'); plt.tight_layout()
plt.show()