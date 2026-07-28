import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# ==========================================
# PHYSICAL AND CHEMICAL PARAMETERS
# ==========================================
# Cell Geometry
A = 1.0          # Plate Area (m^2)
d = 0.05         # Distance between plates (m)
V_sol = A * d    # Electrolyte volume (m^3) -> 50 Liters

# Electrolyte (20% H2SO4 + Copper)
rho_sol = 1140.0 # Density (kg/m^3)
m_sol = rho_sol * V_sol  # Mass of solution (kg) -> ~57 kg
C_p = 3500.0     # Heat capacity of 20% H2SO4 (J/kg/K)
kappa = 70.0     # Electrical conductivity (S/m)

# Solution Resistance
R_sol = d / (kappa * A) # Ohmic resistance (Ohm) -> 0.000714 Ohm

# Parasitic and Contact Resistances (Ohms)
R_contact = 0.0001     # Contact resistance at busbar junctions (0.1 mOhm)
R_plate = 0.00005      # Electrode plate internal resistance (0.05 mOhm)
R_peripheral = 0.00002 # Peripheral busbar & cable resistance (0.02 mOhm)
f_contact = 0.5        # Fraction of contact resistance heat conducted to electrolyte

# Electrochemical parameters
E_eq = 0.89      # Equilibrium cell potential (V)
E_tn = 1.15      # Thermoneutral voltage (V)

# Activation Kinetics & Double-Layer
C_c = 0.2        # Cathode double-layer capacitance (F)
C_a = 0.2        # Anode double-layer capacitance (F)
I0_c = 10.0      # Cathode exchange current (A)
I0_a = 5.0       # Anode exchange current (A)
beta_c = 19.1    # Cathode charge transfer coeff term (V^-1)
beta_a = 19.1    # Anode charge transfer coeff term (V^-1)

# Mass Transport (Copper Diffusion at Cathode)
C_b = 15.74      # Bulk copper concentration (mol/m^3) corresponding to 1 g/L
tau_diff = 15.0  # Diffusion time constant (s)
delta = 1e-4     # Boundary layer thickness (m)
F = 96485.0      # Faraday constant (C/mol)
R_gas = 8.314    # Gas constant (J/mol/K)
T_ref = 303.15   # Reference temperature (30 C, K)

# Heat Transfer
U = 10.0         # Convective heat transfer coefficient (W/m^2/K)
A_loss = 2.05    # Heat loss area (m^2) (top surface + outer plates)
T_amb = 25.0     # Ambient temperature (C)
t_run = 1200.0   # Operation time (20 minutes = 1200 s)

# Default currents
I_avg_target = 300.0 # Target average current (A)
I_peak_target = 300.0 # Target peak current (A)

# ==========================================
# CELL MODEL SIMULATION FUNCTIONS
# ==========================================

def solve_overpotentials(f, D, I_p):
    """
    Finds the periodic steady state overpotentials (cathode & anode activation)
    and returns their average values during the 'on' period.
    """
    T = 1.0 / f
    T_on = D * T
    T_off = T - T_on
    
    # ODE systems for on and off periods
    # y = [eta_c, eta_a]
    def ode_on(t, y):
        # Current is I_p during on-period
        d_eta_c = (I_p - 2 * I0_c * np.sinh(beta_c * y[0])) / C_c
        d_eta_a = (I_p - 2 * I0_a * np.sinh(beta_a * y[1])) / C_a
        return [d_eta_c, d_eta_a]
        
    def ode_off(t, y):
        # Current is 0 during off-period
        d_eta_c = (-2 * I0_c * np.sinh(beta_c * y[0])) / C_c
        d_eta_a = (-2 * I0_a * np.sinh(beta_a * y[1])) / C_a
        return [d_eta_c, d_eta_a]

    # Stiff solver over 5 cycles to reach periodic steady state
    y_start = [0.0, 0.0]
    for _ in range(5):
        sol_on = solve_ivp(ode_on, [0, T_on], y_start, method='BDF', rtol=1e-4, atol=1e-5)
        y_mid = sol_on.y[:, -1]
        sol_off = solve_ivp(ode_off, [0, T_off], y_mid, method='BDF', rtol=1e-4, atol=1e-5)
        y_start = sol_off.y[:, -1]
        
    # Solve 6th cycle to compute average overpotential during on-period
    sol_on = solve_ivp(ode_on, [0, T_on], y_start, method='BDF', rtol=1e-4, atol=1e-5)
    t_on = sol_on.t
    eta_c_on = sol_on.y[0]
    eta_a_on = sol_on.y[1]
    
    # Integrate using trapezoidal rule
    eta_c_avg_on = np.trapz(eta_c_on, t_on) / T_on
    eta_a_avg_on = np.trapz(eta_a_on, t_on) / T_on
    
    return eta_c_avg_on, eta_a_avg_on

def solve_concentration_overpotential_on(f, D, I_p):
    """
    Solves the periodic steady-state concentration overpotential during the 'on' period.
    Uses the analytical solution of the linear diffusion boundary layer ODE.
    """
    T = 1.0 / f
    T_on = D * T
    
    K = 1.0 / (2 * F * A * delta)
    E_1 = np.exp(-T_on / tau_diff)
    E_2 = np.exp(-(T - T_on) / tau_diff)
    
    C_1 = C_b - K * tau_diff * I_p
    
    # Cs at start of cycle Cs(0)
    Cs_0 = C_b - K * tau_diff * I_p * (E_2 * (1.0 - E_1)) / (1.0 - E_1 * E_2 + 1e-15)
    
    # Sample 50 points during on-time
    t_vals = np.linspace(0, T_on, 50)
    Cs_vals = C_1 + (Cs_0 - C_1) * np.exp(-t_vals / tau_diff)
    
    # Cap concentration to avoid negative values
    Cs_min = 1e-4 * C_b
    Cs_vals = np.maximum(Cs_min, Cs_vals)
    
    # Concentration overpotential (magnitude, V)
    eta_conc_vals = -(R_gas * T_ref / (2 * F)) * np.log(Cs_vals / C_b)
    
    # Average concentration overpotential during on-period
    eta_conc_avg_on = np.trapz(eta_conc_vals, t_vals) / T_on
    return eta_conc_avg_on

def get_average_heat_generation(f, D, I_val, mode='avg'):
    """
    Calculates the average heat generation rate (W) for the cell over a cycle.
    mode='avg': I_val is the constant average current, peak current is I_val/D
    mode='peak': I_val is the constant peak current, average current is I_val*D
    """
    # Define peak current (I_p) and average current (I_avg)
    if mode == 'avg':
        I_avg = I_val
        I_p = I_avg / D
    else:
        I_p = I_val
        I_avg = I_p * D
        
    R_extra_heating = R_plate + f_contact * R_contact
        
    # Check for DC limit
    if f == 0.0:
        # DC case: constant current = I_avg
        eta_c = (1.0 / beta_c) * np.arcsinh(I_avg / (2 * I0_c))
        eta_a = (1.0 / beta_a) * np.arcsinh(I_avg / (2 * I0_a))
        
        K = 1.0 / (2 * F * A * delta)
        Cs = np.maximum(1e-4 * C_b, C_b - K * tau_diff * I_avg)
        eta_conc = -(R_gas * T_ref / (2 * F)) * np.log(Cs / C_b)
        
        Q_joule_sol = I_avg**2 * R_sol
        Q_joule_extra = I_avg**2 * R_extra_heating
        Q_joule = Q_joule_sol + Q_joule_extra
        Q_over = I_avg * (eta_a + eta_c + eta_conc)
        Q_chem = I_avg * (E_eq - E_tn)
        
        Q_gen_avg = Q_joule + Q_over + Q_chem
        return Q_gen_avg
        
    # Solve activation overpotentials
    eta_c_on, eta_a_on = solve_overpotentials(f, D, I_p)
    
    # Solve concentration overpotentials
    eta_conc_on = solve_concentration_overpotential_on(f, D, I_p)
    
    # Power and Heat Generation
    Q_joule_sol = D * I_p**2 * R_sol
    Q_joule_extra = D * I_p**2 * R_extra_heating
    Q_joule = Q_joule_sol + Q_joule_extra
    Q_over = I_avg * (eta_c_on + eta_a_on + eta_conc_on)
    Q_chem = I_avg * (E_eq - E_tn)
    
    Q_gen_avg = Q_joule + Q_over + Q_chem
    return Q_gen_avg

def calculate_temp_rise(Q_gen_avg):
    """
    Calculates the temperature rise (delta T, in K or C) after 20 minutes of operation.
    Accounts for convective and radiative heat losses.
    """
    # Transient solution to m_sol * C_p * dT/dt = Q_gen_avg - U * A_loss * (T - T_amb)
    # T(t) - T_amb = (Q_gen_avg / (U * A_loss)) * (1 - exp(- (U * A_loss / (m_sol * C_p)) * t))
    UA = U * A_loss
    mC = m_sol * C_p
    
    if UA == 0:
        # Adiabatic case
        return (Q_gen_avg * t_run) / mC
    else:
        return (Q_gen_avg / UA) * (1.0 - np.exp(-(UA / mC) * t_run))

# ==========================================
# MAIN EXECUTION AND PLOTTING
# ==========================================

if __name__ == "__main__":
    print("Starting simulation...")
    
    # Grid for Frequency and Duty Cycle
    # We avoid f = 0 in grid to prevent division by zero in ODE period,
    # but we can set the lowest frequency to 0.1 Hz and compute DC separately.
    freqs = np.linspace(1.0, 60.0, 20)  # 1 Hz to 60 Hz
    duties = np.linspace(0.05, 1.0, 20)  # 5% to 100% duty cycle
    
    F_grid, D_grid = np.meshgrid(freqs, duties)
    
    # Initialize output arrays
    dT_avg_current = np.zeros_like(F_grid)
    dT_peak_current = np.zeros_like(F_grid)
    
    # Run simulation over grid
    n_points = F_grid.size
    count = 0
    for i in range(len(duties)):
        for j in range(len(freqs)):
            f = F_grid[i, j]
            D = D_grid[i, j]
            
            # Case 1: Constant Average Current (300 A)
            Q_avg = get_average_heat_generation(f, D, I_avg_target, mode='avg')
            dT_avg_current[i, j] = calculate_temp_rise(Q_avg)
            
            # Case 2: Constant Peak Current (300 A)
            Q_peak = get_average_heat_generation(f, D, I_peak_target, mode='peak')
            dT_peak_current[i, j] = calculate_temp_rise(Q_peak)
            
            count += 1
            if count % 50 == 0:
                print(f"Progress: {count}/{n_points} grid points calculated.")
                
    # Also calculate DC (f = 0, D = 1) for reference
    Q_dc_avg = get_average_heat_generation(0.0, 1.0, I_avg_target, mode='avg')
    dT_dc_avg = calculate_temp_rise(Q_dc_avg)
    print(f"\nReference DC (Average 300 A) Temp Rise: {dT_dc_avg:.3f} °C")
    
    # ==========================================
    # GENERATE PLOTS
    # ==========================================
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle('Electrolyte Temperature Rise after 20 Minutes (Acoustic/Electrical Pulsing Model)', fontsize=16, fontweight='bold', color='#1a1a1a')
    
    # Style configuration
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    # Subplot 1: Constant Average Current
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(F_grid, D_grid, dT_avg_current, cmap='magma', edgecolor='none', alpha=0.9)
    ax1.set_title('Constant Average Current (I_avg = 300 A)\n(Higher Joule heating at low duty cycle)', fontsize=12, pad=15)
    ax1.set_xlabel('Pulsing Frequency (Hz)', labelpad=10)
    ax1.set_ylabel('Duty Cycle (D)', labelpad=10)
    ax1.set_zlabel('Temperature Rise (°C)', labelpad=10)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='Temp Rise (°C)')
    
    # Subplot 2: Constant Peak Current
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(F_grid, D_grid, dT_peak_current, cmap='viridis', edgecolor='none', alpha=0.9)
    ax2.set_title('Constant Peak Current (I_peak = 300 A)\n(Lower heat generation at low duty cycle)', fontsize=12, pad=15)
    ax2.set_xlabel('Pulsing Frequency (Hz)', labelpad=10)
    ax2.set_ylabel('Duty Cycle (D)', labelpad=10)
    ax2.set_zlabel('Temperature Rise (°C)', labelpad=10)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='Temp Rise (°C)')
    
    # Optimize layout and save
    plt.tight_layout()
    plot_path = 'temperature_rise_surf_plots.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Surface plots saved successfully to {plot_path}")
    
    # Show key stats
    print(f"Simulation completed. Plot saved in current directory as {os.path.abspath(plot_path)}.")
