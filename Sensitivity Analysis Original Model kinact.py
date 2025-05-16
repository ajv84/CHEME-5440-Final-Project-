import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# === Model parameters ===
kon = 100       # µM^-1·s^-1
koff = 0.2      # s^-1
ksub = 0.016    # µM^-1·s^-1
I0 = 0.5        # µM
S0 = 13.0       # µM

# Initial conditions
y0_original = [0.2, S0, 0.0, I0, 0.0, 0.0]

# Time domain
t_span = (0, 7200)  # seconds
t_eval = np.linspace(*t_span, 800)

# Range of kinact values to test (sensitivity analysis)
kinact_values = [0.0005, 0.0011, 0.005, 0.01]  # s^-1
colors = ['green', 'blue', 'orange', 'red']

# === Sensitivity Analysis ===
plt.figure(figsize=(8, 6))
for i, kinact_val in enumerate(kinact_values):
    
    def original_model_kinact(t, y):
        E, S, P, I, EI, EI_cov = y
        dE = -kon * E * I + koff * EI
        dS = -ksub * E * S
        dP = ksub * E * S
        dI = -kon * E * I + koff * EI
        dEI = kon * E * I - koff * EI - kinact_val * EI
        dEI_cov = kinact_val * EI
        return [dE, dS, dP, dI, dEI, dEI_cov]

   
    sol_sens = solve_ivp(original_model_kinact, t_span, y0_original, t_eval=t_eval, method='LSODA')
    
    # Plot product concentration
    plt.plot(sol_sens.t / 60, sol_sens.y[2], label=f'$k_{{inact}}$ = {kinact_val}', color=colors[i])

# === Plot ===
plt.xlabel('Time (minutes)')
plt.ylabel('Product [P] (µM)')
plt.title('Sensitivity of Product Formation to $k_{inact}$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
