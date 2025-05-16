import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def simulate_product_formation():
    # Fixed parameters
    kon = 100  # µM^-1·s^-1 (same for all)
    E0 = 0.02  # Free enzyme µM
    I0 = 0.05  # Inhibitor µM
    S0 = 13.0  # Substrate µM
    P0 = 0.0   # Product µM

    # Time settings
    t_span = (0, 7200)  # 20 minutes
    t_eval = np.linspace(*t_span, 600)

    # Table S3 drug-specific parameters
    inhibitors = {
        'CI-1033':     {'ksub': 0.028, 'koff': 0.19, 'kinact': 0.011},
        'Dacomitinib': {'ksub': 0.023, 'koff': 1.1,  'kinact': 0.0018},
        'Afatinib':    {'ksub': 0.017, 'koff': 0.3,  'kinact': 0.0024},
        'Neratinib':   {'ksub': 0.016, 'koff': 0.2,  'kinact': 0.0011},
        'CL-387785':   {'ksub': 0.017, 'koff': 18,   'kinact': 0.0020},
        'WZ-4002':     {'ksub': 0.024, 'koff': 23,   'kinact': 0.0049},
    }

    # Prepare plot
    plt.figure(figsize=(10, 6))

    for name, params in inhibitors.items():
        ksub = params['ksub']
        koff = params['koff']
        kinact = params['kinact']

        # Initial conditions
        y0 = [E0, I0, 0.0, 0.0, S0, P0]  # E, I, EI, EI_covalent, S, P

        def model(t, y):
            E, I, EI, EI_cov, S, P = y
            dE = -kon * E * I + koff * EI
            dI = -kon * E * I + koff * EI
            dEI = kon * E * I - koff * EI - kinact * EI
            dEI_cov = kinact * EI
            dS = -ksub * E * S
            dP = ksub * E * S
            return [dE, dI, dEI, dEI_cov, dS, dP]

        sol = solve_ivp(model, t_span, y0, t_eval=t_eval, method='LSODA')
        plt.plot(sol.t / 60, sol.y[5], label=name)  # P is y[5]

    # Final plot formatting
    plt.xlabel('Time (minutes)')
    plt.ylabel('Product Concentration [P] (µM)')
    plt.title('Product Formation over Time for Different EGFR Covalent Inhibitors')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run the simulation
simulate_product_formation()
