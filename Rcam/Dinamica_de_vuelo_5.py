import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps

def J(x):
    # Extraer las variables originales
    u_dot = x[:, 0]
    v_dot = x[:, 1]
    w_dot = x[:, 2]
    p_dot = x[:, 3]
    q_dot = x[:, 4]
    r_dot = x[:, 5]
    theta_dot = x[:, 6]
    psi_dot = x[:, 7]
    phi_dot = x[:, 8]
    u = x[:, 9]
    yaw = x[:, 10]
    phi = x[:, 11]
    gamma = x[:, 12]

    # Extraer las variables de control
    def_A = x[:, 13]
    def_E = x[:, 14]
    def_R = x[:, 15]

    # Nuevas variables
    theta = x[:, 16]
    th1 = x[:, 17]
    th2 = x[:, 18]

    # Fórmula con todas las variables
    return (
        0.04545455 * (u_dot / 9.81)**2 +
        0.04545455 * (v_dot / 9.81)**2 +
        0.04545455 * (w_dot / 9.81)**2 +
        0.04545455 * (p_dot / 100)**2 +
        0.04545455 * (q_dot / 20)**2 +
        0.04545455 * (r_dot / 5)**2 +
        0.04545455 * (theta_dot / 12.87878788)**2 +
        0.04545455 * (psi_dot / 1.897321429)**2 +
        0.04545455 * (phi_dot / 1.897321429)**2 +
        0.25 * ((u - 85) / 85)**2 +
        0.25 * (yaw / 3.141592654)**2 +
        0.04545455 * (phi / 0.785398163)**2 +
        0.04545455 * (gamma / 0.34906585)**2 +
        # Variables de control originales con peso 0.1
        0.1 * def_A**2 +
        0.1 * def_E**2 +
        0.1 * def_R**2 +
        # Nuevas variables con peso 0.1
        0.1 * theta**2 +
        0.1 * (th1)**2 +
        0.1 * (th2)**2
    )


# Configuración del PSO
n_dimensions = 19  # Ahora son 19 variables (16 + 3)

lower_bounds = np.array([
    -50, -50, -50, -200, -50, -20, -30, -5, -5, 0, -np.pi, -np.pi/2, -np.pi/4,
    -np.pi/2, -np.pi/2, -np.pi/2,  # def_A, def_E, def_R
    -np.pi/6, 0.05, 0.05  # theta, th1, th2
])
upper_bounds = np.array([
    50, 50, 50, 200, 50, 20, 30, 5, 5, 170, np.pi, np.pi/2, np.pi/4,
    np.pi/2, np.pi/2, np.pi/2,  # def_A, def_E, def_R
    np.pi/6, 1, 1  # theta, th1, th2
])
bounds = (lower_bounds, upper_bounds)
options = {
    'c1': 0.5,
    'c2': 0.3,
    'w': 0.9
}

# Valores iniciales
valores_iniciales = np.array([
    0,      # u_dot
    0,   # v_dot
    0,      # w_dot
    0,   # p_dot
    0,   # q_dot
    0,      # r_dot
    0,      # theta_dot
    0,      # psi_dot
    0,      # phi_dot
    85,     # u
    0,      # yaw
    0,      # phi
    0,    # gamma
    0,      # def_A
    -0.1,      # def_E
    0,      # def_R
    0.1,    # theta (nueva)
    0.08,   # th1 (nueva)
    0.08    # th2 (nueva)
])

# Crear posiciones iniciales
init_pos = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(50, 19))
init_pos[0] = valores_iniciales

# Empezar PSO
optimizer = ps.single.GlobalBestPSO(
    n_particles=50,
    dimensions=n_dimensions,
    options=options,
    bounds=bounds,
    init_pos=init_pos
)

best_cost, best_position = optimizer.optimize(J, iters=500, verbose=True)

# Visualizar resultados
print(f"\nCosto mínimo: {best_cost:.8f}")
print(f"\nVector de resultados óptimos:")
print(best_position)

# Mostrar todas las variables de control
print("\nCondición de vuelo:")
print(f"Uo = {best_position[9]:.6f}m/s")
print(f"Yaw = {best_position[10]:.6f}°")
print(f"Roll = {best_position[11]:.6f}°") #print phi
print(f"Climb = {best_position[12]:.6f}°") #print gamma

print("\nVARIABLES DE CONTROL ÓPTIMAS:")
print(f"def_A = {best_position[13]:.6f} rad ({np.degrees(best_position[13]):.2f}°)")
print(f"def_E = {best_position[14]:.6f} rad ({np.degrees(best_position[14]):.2f}°)")
print(f"def_R = {best_position[15]:.6f} rad ({np.degrees(best_position[15]):.2f}°)")
print(f"th1 = {best_position[17]:.6f} ")
print(f"th2 = {best_position[18]:.6f} ")