import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# === Shared kinetic parameters ===
kon = 100
koff = 0.2
kinact = 0.0011
ksub = 0.016
I0 = 0.5
S0 = 13.0

# Extended model-specific parameters
k_desulfen = 1e-6
k_sulfin = 1e-5
k_syn_E = 6.89e-6
k_deg_E = 6.89e-6
k_syn_EI_cov = 2.58e-5
k_deg_EI_cov = 2.58e-5
k_int_E = 3.3e-3
k_int_EI = 2.5e-3
k_rec = 1.7e-4

# Time domain
t_span = (0, 7200)
t_eval = np.linspace(*t_span, 800)

# Initial conditions for the extended model
y0_extended = [0.2, S0, 0.0, I0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Sensitivity parameter values
k_sulfen_values = [5e-5, 1.1e-4, 2e-4, 5e-4]
k_turnover_values = [3e-6, 6.89e-6, 1.2e-5, 2.5e-5]
colors = ['green', 'blue', 'orange', 'red']

# Create side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# === Subplot A: Sensitivity to k_sulfen ===
for i, k_sulfen_val in enumerate(k_sulfen_values):
    def extended_model_redox(t, y):
        E, S, P, I, EI, EI_cov, EI_sulfen, EI_sulfin, Eox, E_int, EI_cov_int = y
        dE = -kon * E * I + koff * EI + k_syn_E - k_deg_E * E - k_int_E * E + k_rec * E_int
        dS = -ksub * E * S
        dP = ksub * E * S
        dI = -kon * E * I + koff * EI
        dEI = kon * E * I - koff * EI - kinact * EI - k_sulfen_val * EI + k_desulfen * EI_sulfen
        dEI_cov = kinact * EI + k_syn_EI_cov - k_deg_EI_cov * EI_cov - k_int_EI * EI_cov + k_rec * EI_cov_int
        dEI_sulfen = k_sulfen_val * EI - k_desulfen * EI_sulfen - k_sulfin * EI_sulfen
        dEI_sulfin = k_sulfin * EI_sulfen
        dEox = k_deg_E * EI_sulfen + k_deg_E * EI_sulfin - k_desulfen * EI_sulfen
        dE_int = k_int_E * E - k_rec * E_int
        dEI_cov_int = k_int_EI * EI_cov - k_rec * EI_cov_int
        return [dE, dS, dP, dI, dEI, dEI_cov, dEI_sulfen, dEI_sulfin, dEox, dE_int, dEI_cov_int]

    sol = solve_ivp(extended_model_redox, t_span, y0_extended, t_eval=t_eval, method='LSODA')
    ax1.plot(sol.t / 60, sol.y[2], label=f'$k_{{sulfen}}$ = {k_sulfen_val}', color=colors[i])

ax1.set_title('Sensitivity to $k_{sulfen}$ (Redox Diversion)')
ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('Product [P] (µM)')
ax1.grid(True)
ax1.legend()

# === Subplot B: Sensitivity to Turnover Rate ===
k_sulfen = 1.1e-4  # baseline sulfenylation rate for turnover analysis

for i, k_turnover_val in enumerate(k_turnover_values):
    def extended_model_turnover(t, y):
        E, S, P, I, EI, EI_cov, EI_sulfen, EI_sulfin, Eox, E_int, EI_cov_int = y
        dE = -kon * E * I + koff * EI + k_turnover_val - k_turnover_val * E - k_int_E * E + k_rec * E_int
        dS = -ksub * E * S
        dP = ksub * E * S
        dI = -kon * E * I + koff * EI
        dEI = kon * E * I - koff * EI - kinact * EI - k_sulfen * EI + k_desulfen * EI_sulfen
        dEI_cov = kinact * EI + k_turnover_val - k_turnover_val * EI_cov - k_int_EI * EI_cov + k_rec * EI_cov_int
        dEI_sulfen = k_sulfen * EI - k_desulfen * EI_sulfen - k_sulfin * EI_sulfen
        dEI_sulfin = k_sulfin * EI_sulfen
        dEox = k_turnover_val * EI_sulfen + k_turnover_val * EI_sulfin - k_desulfen * EI_sulfen
        dE_int = k_int_E * E - k_rec * E_int
        dEI_cov_int = k_int_EI * EI_cov - k_rec * EI_cov_int
        return [dE, dS, dP, dI, dEI, dEI_cov, dEI_sulfen, dEI_sulfin, dEox, dE_int, dEI_cov_int]

    sol = solve_ivp(extended_model_turnover, t_span, y0_extended, t_eval=t_eval, method='LSODA')
    ax2.plot(sol.t / 60, sol.y[2], label=f'$k_{{turnover}}$ = {k_turnover_val}', color=colors[i])

ax2.set_title('Sensitivity to Turnover Rate')
ax2.set_xlabel('Time (minutes)')
ax2.grid(True)
ax2.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

