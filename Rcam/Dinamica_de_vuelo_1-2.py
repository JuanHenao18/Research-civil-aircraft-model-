# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 15:18:40 2025

@author: mrest
"""

import numpy as np
import matplotlib.pyplot as plt

def x_dot(u,v,w,p,q,r,phi,theta,psi,def_A,def_E,def_R,th1,th2):   #x_dot función de x_bar , u_bar

  #Define constants

  g = 9.81 #gravity (m/s^2)
  rho = 1.225 #density (kg/m^3)
  MASS = 120000 #aircraft  mass (kg)
  XAPT1 = 0.0
  YAPT1 = -7.94 # (m)
  ZAPT1 = -1.9 #(m)
  XAPT2 = 0.0 #(m)
  YAPT2 = 7.94 #(m)
  ZAPT2 = -1.9 #(m)
  LT = 24.8 #distance between CoG and the aerodynamic center of the tail
  S = 260 # Wing planform area (m^2)
  ST = 64.0 #Tail unit planform area (m^2)
  c_bar = 6.6 #mean aerodynamic chord (m)
  alpha_lift_zero = -11.5*(np.pi/180)
  n = 5.5
  a3 = -768.5
  a2 = 609.2
  a1 = -155.2
  a0 = 15.212
  epsilon_alpha = 0.25

  #Control limits

  if def_A < -25*(np.pi/180):
    def_A = -25*(np.pi/180)
  elif def_A > 25*(np.pi/180):
    def_A = 25*(np.pi/180)

  if def_E < -25*(np.pi/180):
    def_E = -25*(np.pi/180)
  elif def_E > 10*(np.pi/180):
    def_E = 10*(np.pi/180)

  if def_R < -30*(np.pi/180):
    def_R = -30*(np.pi/180)
  elif def_R > 30*(np.pi/180):
    def_R = 30*(np.pi/180)

  if th1 < 0.5*(np.pi/180):
    th1 = 0.5*(np.pi/180)
  elif th1 > 10*(np.pi/180):
    th1 = 10*(np.pi/180)

  if th2 < 0.5*(np.pi/180):
    th2 = 0.5*(np.pi/180)
  elif th2 > 10*(np.pi/180):
    th2 = 10*(np.pi/180)

  #intermediate variables

  V_a = np.sqrt((u**2)+(v**2)+(w**2))

  alpha = np.atan2(w,u)

  betha = np.asin(v/V_a)

  Q = 0.5*rho*(V_a**2)

  w_b = np.array([p,q,r]).T

  Vb = np.array([u,v,w]).T

  #Nondimensional Aero Forces coefficients in Fs

  #Wing body (El problema de este modelo es que no modela la parte negativa)

  if alpha <= 14.5*(np.pi/180):
    CLwb = n*(alpha-alpha_lift_zero)
  else:
    CLwb = a3*(alpha**3) + a2*(alpha**2) + a1*alpha + a0

  #Tail

  downwash = epsilon_alpha*(alpha-alpha_lift_zero)

  alpha_tail = alpha - downwash + def_E + 1.3*q*(LT/V_a)  #(Ultimo termino es porque el cambio de q perturba el flujo de V_inf que llega al tail)

  CL_tail = 3.1*(ST/S)*alpha_tail

  #CL total

  CL = CLwb + CL_tail

  #Drag

  CD = 0.13 + 0.07*(5.5*alpha + 0.654)**2 #(El problema de este modelo es que no esta incluyendo el sideslip)

  #Total side force coefficient

  Cy = -1.6*betha + 0.24*def_R #(Depende del rudder porque a mayor def del rudder mayor la fuerza lateral)

  #Rotate form Fs to Fw (La rotacion es para tener en cuenta el sideslip)

  Cw_s_betha = np.array([[np.cos(betha),np.sin(betha),0],
                [-np.sin(betha),np.cos(betha),0],
                [0,0,1]])

  CF_s = np.array([CD,Cy,CL]).T

  CF_w = np.dot(Cw_s_betha, CF_s)

  #Aerodynamic force in Fb

  L = CL*Q*S

  D = CD*Q*S

  Y = Cy*Q*S

  FA_s = np.array([-D,Y,-L]).T #Vector de fuerzas aerodinamicas

  Cb_s_alpha = np.array([[np.cos(alpha),0,-np.sin(alpha)],
                          [0,1,0],
                          [np.sin(alpha),0,np.cos(alpha)]])

  FA_b = np.dot(Cb_s_alpha,FA_s)

  #Nondimensional Aero Moment Coefficient about AC in Fb (tres porque son tres ejes)

  n_bar = np.array([-1.4*betha,-0.59-(3.1*(ST*LT)/(S*c_bar)*(alpha-downwash)),(1-alpha*(180/15*np.pi))*betha]).T
  Cm_x = (c_bar/V_a)*np.array([[-11,0,5],
                    [0,-4.03*((ST*LT**2)/(S*c_bar**2)),0],
                    [1.7,0,-11.5]])
  Cm_u = np.array([[-0.6,0,0.22],
                   [0,(-3.1*(ST*LT)/(S*c_bar)),0],
                   [0,0,-0.63]])
  CM_ac_b = n_bar + np.dot(Cm_x, w_b) + np.dot(Cm_u, np.array([def_A, def_E, def_R]).T)
  Cl_ac, Cm_ac, Cn_ac = CM_ac_b

  #Aero moment about AC in Fb

  MAac_b = CM_ac_b*Q*S*c_bar

  #Aero moment about cg in Fb

  r_cg = np.array([0.23*c_bar,0,0.1*c_bar]).T
  r_ac = np.array([0.12*c_bar,0,0]).T
  MAcg_b = MAac_b + np.cross(FA_b,(r_cg-r_ac))

  #Propulsion effects PREGUNTAR

  F1 = th1*MASS*g
  F2 = th2*MASS*g

  FE1_b = np.array([F1,0,0]).T
  FE2_b = np.array([F2,0,0]).T
  FE_b = FE1_b + FE2_b

  miu1_b = np.array([0.23*c_bar-XAPT1,YAPT1,0.1*c_bar-ZAPT1]).T
  miu2_b = np.array([0.23*c_bar-XAPT2,YAPT2,0.1*c_bar-ZAPT2]).T

  MEcg1_b = np.cross(miu1_b,FE1_b)
  MEcg2_b = np.cross(miu2_b,FE2_b)

  MEcg_b = MEcg1_b + MEcg2_b

  #Gravity effects

  Fg_b = np.array([-g*np.sin(theta),g*np.cos(theta)*np.sin(psi),g*np.cos(theta)*np.cos(psi)]).T * MASS

  #Explicit first order form

  F_b = Fg_b + FE_b + FA_b
  Mcg_b = MEcg_b + MAcg_b

  I_b = MASS*np.array([[40.07,0,-2.0923],
                  [0,64,0],
                  [-2.0923,0,99.92]])

  Rinv = np.array([[1,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],
                   [0,np.cos(phi),-np.sin(phi)],
                   [0,np.sin(phi)/np.cos(theta),np.cos(phi)/np.cos(theta)]])

  vel_dot = np.array((1/MASS)*F_b - np.cross(w_b,Vb))
  angular_dot = np.array(np.dot(np.linalg.inv(I_b), (Mcg_b-np.cross(w_b,np.dot(I_b, w_b)))))
  euler_dot =  np.array(np.dot(Rinv, w_b))

  u_dot,v_dot,w_dot = vel_dot
  p_dot,q_dot,r_dot = angular_dot
  phi_dot,theta_dot,psi_dot = euler_dot

  x_dot_vec = np.array([u_dot,v_dot,w_dot,p_dot,q_dot,r_dot,phi_dot,theta_dot,psi_dot,def_A,def_E,def_R,th1,th2]).T

  return x_dot_vec

# Paso de tiempo y duración
dt = 1e-3
t_final = 180 # 3 minutos
N = int(t_final / dt)

# Estado inicial
x = np.array([85, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, -0.1, 0, 0.08, 0.08])

X_hist = np.zeros((N+1, len(x)))
time = np.linspace(0, t_final, N+1)
X_hist[0, :] = x

for k in range(N):
    x_dot_vec = x_dot(*x)

    x[:9] = x[:9] + x_dot_vec[:9] * dt # x(t+dt) = x(t)+̇x(t)dt de u a psi

    X_hist[k+1, :] = x

labels = ['u','v','w','p','q','r','phi','theta','psi','def_A','def_E','def_R','th1','th2']

# Longitudinal dynamics
plt.figure(figsize=(10,6))
plt.subplot(2,2,1); plt.plot(time, X_hist[:,0]); plt.ylabel('u [m/s]'); plt.grid(True)
plt.subplot(2,2,2); plt.plot(time, X_hist[:,2]); plt.ylabel('w [m/s]'); plt.grid(True)
plt.subplot(2,2,3); plt.plot(time, X_hist[:,4]); plt.ylabel('q [rad/s]'); plt.grid(True)
plt.subplot(2,2,4); plt.plot(time, X_hist[:,7]); plt.ylabel('theta [rad]'); plt.grid(True)
plt.suptitle('Longitudinal motion'); plt.tight_layout()
plt.show()

# Lateral-directional
plt.figure(figsize=(10,6))
plt.subplot(3,2,1); plt.plot(time, X_hist[:,1]); plt.ylabel('v [m/s]'); plt.grid(True)
plt.subplot(3,2,2); plt.plot(time, X_hist[:,3]); plt.ylabel('p [rad/s]'); plt.grid(True)
plt.subplot(3,2,3); plt.plot(time, X_hist[:,5]); plt.ylabel('r [rad/s]'); plt.grid(True)
plt.subplot(3,2,4); plt.plot(time, X_hist[:,6]); plt.ylabel('phi [rad]'); plt.grid(True)
plt.subplot(3,2,5); plt.plot(time, X_hist[:,8]); plt.ylabel('psi [rad]'); plt.grid(True)
plt.suptitle('Lateral-directional motion'); plt.tight_layout()
plt.show()