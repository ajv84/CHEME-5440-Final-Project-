import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def egfr_covalent_neratinib_original():
    # === Rate constants (Neratinib) ===
    kon = 100          # µM^-1·s^-1
    koff = 0.2         # s^-1
    kinact = 0.0011    # s^-1
    ksub = 0.016       # µM^-1·s^-1

    # === Initial concentrations (µM) ===
    E0 = 0.02         # Free enzyme (20 nM)
    I0 = 0.05         # Inhibitor (50 nM)
    EI0 = 0.0         # Reversible complex
    EI_covalent0 = 0.0  # Covalent complex
    S0 = 13.0         # Substrate
    P0 = 0.0          # Product

    y0 = [E0, I0, EI0, EI_covalent0, S0, P0]

    # === Time span ===
    t_span = (0, 7200)  # seconds
    t_eval = np.linspace(*t_span, 600)

    # === ODE system ===
    def model(t, y):
        E, I, EI, EI_cov, S, P = y
        dE = -kon * E * I + koff * EI
        dI = -kon * E * I + koff * EI
        dEI = kon * E * I - koff * EI - kinact * EI
        dEI_cov = kinact * EI
        dS = -ksub * E * S
        dP = ksub * E * S
        return [dE, dI, dEI, dEI_cov, dS, dP]

    # === Solve system ===
    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='LSODA')

    # === Plot results ===
    plt.figure(figsize=(10, 6))
    plt.plot(sol.t / 60, sol.y[0], label='Free Enzyme [E]', color='blue')
    plt.plot(sol.t / 60, sol.y[2], label='Reversible Complex [EI]', color='red')
    plt.plot(sol.t / 60, sol.y[3], label='Covalent Complex [E–I]', color='black')
    plt.plot(sol.t / 60, sol.y[5], label='Product [P]', color='green')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Concentration (µM)')
    plt.title('EGFR Covalent Inhibition by Neratinib')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run
egfr_covalent_neratinib_original()
