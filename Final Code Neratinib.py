import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def egfr_full_reconstructed_model():
    # === Kinetic parameters ===
    kon = 100                 # µM^-1·s^-1
    koff = 0.2                # s^-1
    kinact = 0.0011           # s^-1
    ksub = 0.016              # µM^-1·s^-1

    # === Redox rates (sulfenylation/sulfinylation) ===
    k_sulfen = 1.1e-4         # EI → EI_sulfen
    k_desulfen = 1e-6         # EI_sulfen → EI
    k_sulfin = 1e-5           # EI_sulfen → EI_sulfin

    # === Turnover ===
    k_syn_E = 6.89e-6         # µM/s
    k_deg_E = 6.89e-6         # s^-1
    k_syn_EI_cov = 2.58e-5    # µM/s
    k_deg_EI_cov = 2.58e-5    # s^-1

    # === Internalization & recycling ===
    k_int_E = 3.3e-3          # s^-1
    k_int_EI = 2.5e-3         # s^-1
    k_rec = 1.7e-4            # s^-1

    # === Initial concentrations ===
    E0 = 0.2        # Free EGFR (µM)
    S0 = 13.0       # Substrate (µM)
    P0 = 0.0        # Product (µM)
    I0 = 0.5        # Inhibitor (µM)
    EI0 = 0.0       # Reversible complex
    EI_cov0 = 0.0   # Covalent adduct
    EI_sulfen0 = 0.0
    EI_sulfin0 = 0.0
    Eox0 = 0.0
    E_int0 = 0.0
    EI_cov_int0 = 0.0

    y0 = [E0, S0, P0, I0, EI0, EI_cov0, EI_sulfen0, EI_sulfin0, Eox0, E_int0, EI_cov_int0]

    t_span = (0, 7200)
    t_eval = np.linspace(*t_span, 800)

    def model(t, y):
        E, S, P, I, EI, EI_cov, EI_sulfen, EI_sulfin, Eox, E_int, EI_cov_int = y

        dE = -kon * E * I + koff * EI + k_syn_E - k_deg_E * E \
             - k_int_E * E + k_rec * E_int
             
        dS = -ksub * E * S
        
        dP = ksub * E * S
        
        dI = -kon * E * I + koff * EI

        dEI = kon * E * I - koff * EI - kinact * EI \
              - k_sulfen * EI + k_desulfen * EI_sulfen

        dEI_cov = kinact * EI + k_syn_EI_cov - k_deg_EI_cov * EI_cov \
                  - k_int_EI * EI_cov + k_rec * EI_cov_int

        dEI_sulfen = k_sulfen * EI - k_desulfen * EI_sulfen - k_sulfin * EI_sulfen
        
        dEI_sulfin = k_sulfin * EI_sulfen
        
        dEox = k_deg_E * EI_sulfen + k_deg_E * EI_sulfin - k_desulfen * EI_sulfen

        dE_int = k_int_E * E - k_rec * E_int
        
        dEI_cov_int = k_int_EI * EI_cov - k_rec * EI_cov_int

        return [dE, dS, dP, dI, dEI, dEI_cov, dEI_sulfen, dEI_sulfin, Eox, dE_int, dEI_cov_int]

    sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='LSODA')

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(sol.t / 60, sol.y[2], label='Product [P]', color='purple', linewidth=2)
    plt.plot(sol.t / 60, sol.y[0] + sol.y[4], label='Active EGFR (E + EI)', color='green')
    plt.plot(sol.t / 60, sol.y[5], label='Covalent Adduct (E–I)', color='red')
    plt.plot(sol.t / 60, sol.y[6], label='Sulfenylated EI', linestyle='--', color='orange')
    plt.plot(sol.t / 60, sol.y[7], label='Sulfinylated EI', linestyle='--', color='blue')
    plt.plot(sol.t / 60, sol.y[8], label='Oxidized EGFR', linestyle=':', color='black')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Concentration (µM)')
    plt.title('EGFR Inhibition by Neratinib with Redox and Turnover Dynamics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

egfr_full_reconstructed_model()
