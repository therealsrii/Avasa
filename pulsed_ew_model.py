import math
import numpy as np

# Backward compatibility helper for NumPy 2.x trapezoid / trapz
trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

# ==============================================================================
# PHYSICAL CONSTANTS
# ==============================================================================
F = 96485.0       # Faraday constant (C/mol)
R_gas = 8.314     # Universal gas constant (J/mol/K)
M_cu = 63.546     # Molecular weight of Copper (g/mol)
Z_cu = 2.0        # Charge number of copper ion (Cu2+)

# ==============================================================================
# PHYSICAL PROPERTIES CALCULATOR
# ==============================================================================
def electrolyte_properties(temp_C):
    """
    Calculates temperature-dependent dynamic viscosity, density, and Cu diffusion coefficient.
    
    Parameters:
    temp_C (float): Temperature in degrees Celsius.
    
    Returns:
    tuple: (viscosity_mu [Pa*s], density_rho [kg/m^3], kinematic_visc_nu [m^2/s], diffusion_D [m^2/s])
    """
    T_K = temp_C + 273.15
    
    # Dynamic viscosity of water as an approximation for electrolyte viscosity (Pa*s)
    mu = 2.414e-5 * math.pow(10, (247.8 / (T_K - 140.0)))
    
    # Density of 20% H2SO4 solution (kg/m^3)
    rho = 1000.0 * (1.0 - ((temp_C + 288.9414) / (508929.2 * (temp_C + 68.12963))) * (temp_C - 3.9863)**2)
    
    # Kinematic viscosity (m^2/s)
    nu = mu / rho
    
    # Diffusion coefficient of Copper ions (Stokes-Einstein relation with temperature correction) (m^2/s)
    # Reference value: 6.0e-10 m^2/s at 25 C (298.15 K) in water
    D_cu = 6.0e-10 * (T_K / 298.15) * (8.90e-4 / mu)
    
    return mu, rho, nu, D_cu

# ==============================================================================
# MASS TRANSFER COEFFICIENTS & TIME CONSTANTS
# ==============================================================================
def mass_transfer_coefficients(flow_rate_L_min, temp_C, geom=None):
    """
    Calculates mass transfer coefficients, boundary thicknesses, and time constants.
    
    Parameters:
    flow_rate_L_min (float): Electrolyte flow rate in L/min.
    temp_C (float): Temperature in degrees Celsius.
    geom (dict, optional): Geometry details. If None, defaults are used.
    
    Returns:
    dict: Dictionary containing k, delta, and tau for both jet and dead zones.
    """
    if geom is None:
        geom = {
            'plate_width': 1.0,
            'plate_length': 1.0,
            'd_hole': 0.01,
            'pitch': 0.02,
            'n_faces': 160
        }
        
    plate_width = geom['plate_width']
    plate_length = geom['plate_length']
    d_hole = geom['d_hole']
    pitch = geom['pitch']
    n_faces = geom['n_faces']
    
    mu, rho, nu, D_cu = electrolyte_properties(temp_C)
    
    # Electrode geometry
    n_holes_per_face = (plate_width / pitch) * (plate_length / pitch)
    a_jet_single = math.pi * (d_hole / 2.0)**2
    
    # Fluid dynamics strictly inside jet holes
    flow_m3_s = flow_rate_L_min / (1000.0 * 60.0)
    v_jet = flow_m3_s / (n_holes_per_face * a_jet_single)
    
    # Re and Sc numbers
    Re_jet = (v_jet * d_hole) / nu if nu > 0 else 0.0
    Sc = nu / D_cu if D_cu > 0 else 0.0
    
    # Mass transfer coefficient in jet zones (Sherwood relation Sh = 1.0 * Re^0.5 * Sc^0.33)
    Sh_jet = 1.0 * math.sqrt(max(0.1, Re_jet)) * math.pow(Sc, 0.33)
    k_jet = (Sh_jet * D_cu) / d_hole
    
    # Dead zone mass transfer coefficient is a fraction of the jet zone
    k_dead = k_jet * 0.15
    
    # Boundary layer thicknesses delta = D_cu / k (m)
    delta_jet = D_cu / k_jet
    delta_dead = D_cu / k_dead
    
    # Diffusion time constants tau = delta / k = D_cu / k^2 (s)
    tau_jet = D_cu / (k_jet**2)
    tau_dead = D_cu / (k_dead**2)
    
    return {
        'k_jet': k_jet,
        'k_dead': k_dead,
        'delta_jet': delta_jet,
        'delta_dead': delta_dead,
        'tau_jet': tau_jet,
        'tau_dead': tau_dead,
        'D_cu': D_cu,
        'n_holes_per_face': n_holes_per_face,
        'a_jet_single': a_jet_single
    }

# ==============================================================================
# PULSED ELECTROWINNING CORE SOLVER
# ==============================================================================
def solve_pulsed_zone_efficiency(J_peak, J_lim, tau, T_on, T_off, C_bulk, temp_C):
    """
    Analytically solves transient concentration depletion for a single zone (jet or dead)
    under pulsed current and returns the zone current efficiency and average values.
    
    Returns:
    tuple: (efficiency [0-1], C_s_avg_on [g/L], eta_conc_avg_on [V])
    """
    T_K = temp_C + 273.15
    
    # If no current, efficiency is 100% (non-operational) and concentration is bulk
    if J_peak <= 0.01:
        return 1.0, C_bulk, 0.0
        
    # Duty cycle decay constants
    E_1 = math.exp(-min(50.0, T_on / tau))
    E_2 = math.exp(-min(50.0, T_off / tau))
    
    # Steady state concentration under constant peak current (can be negative if J_peak > J_lim)
    # J_lim = z * F * k * C_bulk  =>  J_peak / (z * F * k) = C_bulk * (J_peak / J_lim)
    C_ss_on = C_bulk * (1.0 - J_peak / J_lim)
    
    # Case 1: Peak current is below limiting current. No depletion possible.
    if J_peak <= J_lim:
        # PSS start-of-on-pulse concentration Cs(0)
        Cs_0 = C_bulk - (C_bulk - C_ss_on) * (E_2 * (1.0 - E_1)) / (1.0 - E_1 * E_2 + 1e-15)
        Cs_0 = max(1e-4 * C_bulk, Cs_0)
        
        # Profile during 'on' time: Cs(t) = C_ss_on + (Cs_0 - C_ss_on) * exp(-t/tau)
        # Integrate Cs(t) analytically to get average Cs during 'on' time
        C_s_avg_on = C_ss_on + (tau / T_on) * (Cs_0 - C_ss_on) * (1.0 - E_1)
        C_s_avg_on = max(1e-4 * C_bulk, C_s_avg_on)
        
        # Numerical integration of concentration overpotential to be precise
        t_vals = np.linspace(0, T_on, 50)
        Cs_vals = C_ss_on + (Cs_0 - C_ss_on) * np.exp(-t_vals / tau)
        Cs_vals = np.maximum(1e-4 * C_bulk, Cs_vals)
        eta_conc_vals = -(R_gas * T_K / (Z_cu * F)) * np.log(Cs_vals / C_bulk)
        eta_conc_avg_on = trapezoid(eta_conc_vals, t_vals) / T_on
        
        return 1.0, C_s_avg_on, eta_conc_avg_on
        
    # Case 2: Peak current exceeds limiting current. Depletion to Cs = 0 is possible.
    else:
        # First, assume depletion occurs during the on-time.
        # Under depletion, Cs(T_on) = 0.
        # During off-time, relaxation occurs starting from Cs(T_on) = 0:
        # Cs(0) = C_bulk * (1 - E_2)
        Cs_0_dep = C_bulk * (1.0 - E_2)
        
        # Calculate time required to deplete surface concentration to zero:
        # 0 = C_ss_on + (Cs_0_dep - C_ss_on) * exp(-t_deplete/tau)
        # exp(-t_deplete/tau) = -C_ss_on / (Cs_0_dep - C_ss_on) = C_ss_on / (C_ss_on - Cs_0_dep)
        ratio = C_ss_on / (C_ss_on - Cs_0_dep)
        
        if ratio > 0:
            t_deplete = -tau * math.log(ratio)
        else:
            t_deplete = 0.0 # Instant depletion
            
        # Check if depletion happens within the pulse duration
        if t_deplete < T_on:
            # Depletion actually occurs!
            # Charge transferred during on-pulse per unit area:
            # q_act = J_peak * t_deplete + J_lim * (T_on - t_deplete)
            q_act = J_peak * t_deplete + J_lim * (T_on - t_deplete)
            q_app = J_peak * T_on
            efficiency = q_act / q_app
            
            # Average concentration during 'on' time:
            # Integrated from 0 to t_deplete, and 0 from t_deplete to T_on
            C_s_avg_on = (1.0 / T_on) * (C_ss_on * t_deplete + tau * Cs_0_dep)
            C_s_avg_on = max(1e-4 * C_bulk, C_s_avg_on)
            
            # Concentration overpotential integration
            t_vals = np.linspace(0, T_on, 100)
            Cs_vals = np.zeros_like(t_vals)
            # Before depletion
            mask_before = t_vals < t_deplete
            Cs_vals[mask_before] = C_ss_on + (Cs_0_dep - C_ss_on) * np.exp(-t_vals[mask_before] / tau)
            # After depletion (cap to small value)
            Cs_vals[~mask_before] = 1e-4 * C_bulk
            Cs_vals = np.maximum(1e-4 * C_bulk, Cs_vals)
            
            eta_conc_vals = -(R_gas * T_K / (Z_cu * F)) * np.log(Cs_vals / C_bulk)
            eta_conc_avg_on = trapezoid(eta_conc_vals, t_vals) / T_on
            
            return efficiency, C_s_avg_on, eta_conc_avg_on
            
        else:
            # Depletion does not happen within the pulse duration (pulse too short)
            # Revert to standard PSS formulas
            Cs_0 = C_bulk - (C_bulk - C_ss_on) * (E_2 * (1.0 - E_1)) / (1.0 - E_1 * E_2 + 1e-15)
            Cs_0 = max(1e-4 * C_bulk, Cs_0)
            
            C_s_avg_on = C_ss_on + (tau / T_on) * (Cs_0 - C_ss_on) * (1.0 - E_1)
            C_s_avg_on = max(1e-4 * C_bulk, C_s_avg_on)
            
            t_vals = np.linspace(0, T_on, 50)
            Cs_vals = C_ss_on + (Cs_0 - C_ss_on) * np.exp(-t_vals / tau)
            Cs_vals = np.maximum(1e-4 * C_bulk, Cs_vals)
            eta_conc_vals = -(R_gas * T_K / (Z_cu * F)) * np.log(Cs_vals / C_bulk)
            eta_conc_avg_on = trapezoid(eta_conc_vals, t_vals) / T_on
            
            return 1.0, C_s_avg_on, eta_conc_avg_on

# ==============================================================================
# PULSED ELECTROWINNING PERFORMANCE METRICS
# ==============================================================================
def calculate_pulsed_ew_metrics(J_avg, f_pulse, D_pulse, flow_rate_L_min, temp_C, C_in_g_L, geom=None):
    """
    Computes overall current efficiency, copper production rate, and outlet concentration under pulsed current.
    
    Parameters:
    J_avg (float): Average applied current density (A/m^2).
    f_pulse (float): Pulsing frequency in Hz. If 0, models continuous DC.
    D_pulse (float): Duty cycle (0 to 1].
    flow_rate_L_min (float): Electrolyte flow rate in L/min.
    temp_C (float): Electrolyte temperature in Celsius.
    C_in_g_L (float): Inlet copper concentration in g/L.
    geom (dict, optional): Cell geometry settings.
    
    Returns:
    dict: Performance metrics including efficiency, production rate, and outlet concentration.
    """
    if geom is None:
        geom = {
            'plate_width': 1.0,
            'plate_length': 1.0,
            'd_hole': 0.01,
            'pitch': 0.02,
            'n_faces': 160
        }
        
    plate_width = geom['plate_width']
    plate_length = geom['plate_length']
    n_faces = geom['n_faces']
    
    plate_area = plate_width * plate_length
    
    # Handle inputs
    if flow_rate_L_min <= 0.01: flow_rate_L_min = 0.01
    if J_avg <= 0.01: J_avg = 0.01
    if C_in_g_L <= 0.0: C_in_g_L = 0.001
    
    # Get mass transfer details
    mt = mass_transfer_coefficients(flow_rate_L_min, temp_C, geom)
    
    n_holes = mt['n_holes_per_face']
    a_jet = mt['a_jet_single']
    
    # Areas (total across all plates in parallel/series depending on flow configuration)
    # Assumes n_faces active surfaces
    A_jet_total = n_faces * n_holes * a_jet
    A_dead_total = n_faces * (plate_area - (n_holes * a_jet))
    A_total = A_jet_total + A_dead_total
    
    # Bulk concentration in mol/m^3
    C_bulk = (C_in_g_L / M_cu) * 1000.0
    
    # Limiting current densities (A/m^2)
    J_lim_jet = Z_cu * F * mt['k_jet'] * C_bulk
    J_lim_dead = Z_cu * F * mt['k_dead'] * C_bulk
    
    # Continuous DC case
    if f_pulse == 0.0 or D_pulse >= 1.0:
        J_actual_jet = min(J_avg, J_lim_jet)
        J_actual_dead = min(J_avg, J_lim_dead)
        
        I_app_total = J_avg * A_total
        I_actual_total = (J_actual_jet * A_jet_total) + (J_actual_dead * A_dead_total)
        efficiency = (I_actual_total / I_app_total) * 100.0
        
        # Average surface concentrations
        C_s_jet_avg = max(1e-4 * C_bulk, C_bulk * (1.0 - J_actual_jet / J_lim_jet))
        C_s_dead_avg = max(1e-4 * C_bulk, C_bulk * (1.0 - J_actual_dead / J_lim_dead))
        
        # Concentration overpotentials
        T_K = temp_C + 273.15
        eta_conc_jet = -(R_gas * T_K / (Z_cu * F)) * math.log(C_s_jet_avg / C_bulk)
        eta_conc_dead = -(R_gas * T_K / (Z_cu * F)) * math.log(C_s_dead_avg / C_bulk)
        
    # Pulsed Current Case
    else:
        T = 1.0 / f_pulse
        T_on = D_pulse * T
        T_off = T - T_on
        
        # Peak applied current density
        J_peak = J_avg / D_pulse
        
        # Jet Zone pulsed efficiency and concentration
        eff_jet, C_s_jet_avg, eta_conc_jet = solve_pulsed_zone_efficiency(
            J_peak, J_lim_jet, mt['tau_jet'], T_on, T_off, C_bulk, temp_C
        )
        
        # Dead Zone pulsed efficiency and concentration
        eff_dead, C_s_dead_avg, eta_conc_dead = solve_pulsed_zone_efficiency(
            J_peak, J_lim_dead, mt['tau_dead'], T_on, T_off, C_bulk, temp_C
        )
        
        # Average actual current during 'on' pulse:
        J_act_on_jet = J_peak * eff_jet
        J_act_on_dead = J_peak * eff_dead
        
        # Average actual current overall:
        J_act_avg_jet = J_act_on_jet * D_pulse
        J_act_avg_dead = J_act_on_dead * D_pulse
        
        I_app_total = J_avg * A_total
        I_actual_total = (J_act_avg_jet * A_jet_total) + (J_act_avg_dead * A_dead_total)
        efficiency = (I_actual_total / I_app_total) * 100.0
        
    # Copper production rate (g/min)
    copper_g_min_theoretical = (I_actual_total * M_cu * 60.0) / (Z_cu * F)
    
    # Mass conservation cap: cannot deposit more copper than flows into the cell
    mass_in_g_min = flow_rate_L_min * C_in_g_L
    copper_g_min = min(copper_g_min_theoretical, mass_in_g_min)
    
    # Re-adjust actual efficiency if capped by feed flow limit
    if copper_g_min < copper_g_min_theoretical:
        I_actual_total = (copper_g_min * Z_cu * F) / (M_cu * 60.0)
        efficiency = (I_actual_total / I_app_total) * 100.0
        
    # Outlet concentration (g/L)
    mass_out_g_min = mass_in_g_min - copper_g_min
    C_out_g_L = mass_out_g_min / flow_rate_L_min
    
    return {
        'efficiency_percent': efficiency,
        'copper_g_min': copper_g_min,
        'C_out_g_L': C_out_g_L,
        'C_s_jet_g_L': (C_s_jet_avg / 1000.0) * M_cu,
        'C_s_dead_g_L': (C_s_dead_avg / 1000.0) * M_cu,
        'eta_conc_jet': eta_conc_jet,
        'eta_conc_dead': eta_conc_dead,
        'I_applied': I_app_total,
        'I_actual': I_actual_total,
        'J_lim_jet': J_lim_jet,
        'J_lim_dead': J_lim_dead
    }

# ==============================================================================
# CONCENTRATION GRADIENT PROFILE SIMULATION (MULTI-PLATE FLOW)
# ==============================================================================
def calculate_pulsed_gradient_profile(n_plates, flow_rate_L_min, J_avg, f_pulse, D_pulse, temp_C, C_in_g_L, geom=None):
    """
    Models sequential copper depletion across a multi-plate container layout under pulsed current.
    
    Parameters:
    n_plates (int): Number of plates in series flow.
    flow_rate_L_min (float): Electrolyte flow rate in L/min.
    J_avg (float): Average applied current density (A/m^2).
    f_pulse (float): Pulsing frequency in Hz.
    D_pulse (float): Duty cycle (0 to 1].
    temp_C (float): Temperature in Celsius.
    C_in_g_L (float): Inlet copper concentration in g/L.
    geom (dict, optional): Cell geometry settings.
    
    Returns:
    list: List of concentration values (g/L) at each stage.
    """
    if geom is None:
        geom = {
            'plate_width': 1.0,
            'plate_length': 1.0,
            'd_hole': 0.01,
            'pitch': 0.02,
            'n_faces': 2 # Each plate in series has 2 active faces
        }
        
    concentrations = [C_in_g_L]
    current_c = C_in_g_L
    
    for _ in range(n_plates):
        res = calculate_pulsed_ew_metrics(
            J_avg, f_pulse, D_pulse, flow_rate_L_min, temp_C, current_c, geom
        )
        current_c = res['C_out_g_L']
        concentrations.append(current_c)
        
    return concentrations

# ==============================================================================
# HYDRAULIC PRESSURE DROP MODELS
# ==============================================================================
def calculate_pressure_drop_orifice(flow_rate_L_min, temp_C, geom=None, K_loss=2.0):
    """
    Calculates total pressure drop across the EW cell using the Orifice Model.
    
    Returns:
    float: Total pressure drop in Pascals (Pa)
    """
    if geom is None:
        geom = {
            'plate_width': 1.0,
            'plate_length': 1.0,
            'd_hole': 0.01,
            'pitch': 0.02,
            'n_faces': 160
        }
        
    plate_width = geom['plate_width']
    plate_length = geom['plate_length']
    d_hole = geom['d_hole']
    pitch = geom['pitch']
    n_faces = geom['n_faces']
    
    _, rho_water, _, _ = electrolyte_properties(temp_C)
    
    n_plates = n_faces / 2.0
    A_cross_section = plate_width * plate_length
    
    # Open area fraction
    n_holes_per_plate = (plate_width / pitch) * (plate_length / pitch)
    a_hole_single = math.pi * (d_hole / 2.0)**2
    total_hole_area = n_holes_per_plate * a_hole_single
    phi = total_hole_area / A_cross_section
    
    # Fluid velocities
    flow_m3_s = flow_rate_L_min / (1000.0 * 60.0)
    v_superficial = flow_m3_s / A_cross_section
    v_hole = v_superficial / phi
    
    # Pressure drop per plate
    delta_p_per_plate = K_loss * 0.5 * rho_water * (v_hole**2)
    delta_p_total = n_plates * delta_p_per_plate
    
    return delta_p_total

def calculate_pressure_drop_ergun(flow_rate_L_min, temp_C, geom=None, plate_thickness_m=0.005, plate_spacing_m=0.05):
    """
    Calculates total pressure drop across the EW cell using the Modified Ergun Equation.
    
    Returns:
    float: Total pressure drop in Pascals (Pa)
    """
    if geom is None:
        geom = {
            'plate_width': 1.0,
            'plate_length': 1.0,
            'd_hole': 0.01,
            'pitch': 0.02,
            'n_faces': 160
        }
        
    plate_width = geom['plate_width']
    plate_length = geom['plate_length']
    d_hole = geom['d_hole']
    pitch = geom['pitch']
    n_faces = geom['n_faces']
    
    mu_water, rho_water, _, _ = electrolyte_properties(temp_C)
    
    n_plates = n_faces / 2.0
    A_cross_section = plate_width * plate_length
    L_total = n_plates * (plate_thickness_m + plate_spacing_m)
    
    # Open area fraction
    n_holes_per_plate = (plate_width / pitch) * (plate_length / pitch)
    a_hole_single = math.pi * (d_hole / 2.0)**2
    total_hole_area = n_holes_per_plate * a_hole_single
    phi = total_hole_area / A_cross_section
    
    # Equivalent porosity and particle diameter
    epsilon_eq = (phi * plate_thickness_m + plate_spacing_m) / (plate_thickness_m + plate_spacing_m)
    a_v = (2.0 * (1.0 - phi) + (4.0 * phi * plate_thickness_m) / d_hole) / (plate_thickness_m + plate_spacing_m)
    Dp_eq = (6.0 * (1.0 - epsilon_eq)) / a_v
    
    # Superficial velocity
    flow_m3_s = flow_rate_L_min / (1000.0 * 60.0)
    v_superficial = flow_m3_s / A_cross_section
    
    # Ergun Terms
    viscous_term = 150.0 * ((mu_water * v_superficial) / (Dp_eq**2)) * (((1.0 - epsilon_eq)**2) / (epsilon_eq**3))
    inertial_term = 1.75 * ((rho_water * (v_superficial**2)) / Dp_eq) * ((1.0 - epsilon_eq) / (epsilon_eq**3))
    
    delta_p_per_m = viscous_term + inertial_term
    delta_p_total = delta_p_per_m * L_total
    
    return delta_p_total

# ==============================================================================
# SOLVENT EXTRACTION (SX) CIRCUIT DESIGN
# ==============================================================================
def calculate_sx_circuit(c_in, q_out, c_out, ketoxime_pct=10.0, naoh_sol_conc=200.0):
    """
    Calculates flow rates for a Copper SX circuit supporting upstream EW operations.
    """
    cu_mass_flow = q_out * c_out
    q_feed_in = cu_mass_flow / c_in
    
    loading_capacity = (ketoxime_pct * 0.55) * 0.8  # Safety factor 0.8
    q_organic = cu_mass_flow / loading_capacity
    
    naoh_mass_needed = cu_mass_flow * (80.0 / 63.54)
    q_naoh = naoh_mass_needed / naoh_sol_conc
    q_acid_strip = q_out
    
    return {
        'cu_mass_flow_g_min': cu_mass_flow,
        'q_feed_in_L_min': q_feed_in,
        'q_organic_L_min': q_organic,
        'q_naoh_L_min': q_naoh,
        'q_acid_strip_L_min': q_acid_strip
    }

# ==============================================================================
# PULSED CURRENT HEAT GENERATION MODEL
# ==============================================================================
def calculate_heat_generation(J_avg, f_pulse, D_pulse, temp_C, C_in_g_L, geom=None, electrical_params=None, flow_rate_L_min=150.0):
    """
    Calculates the individual components of average heat generation rate (W) in the cell.
    
    Each active cathode face (n_faces) faces one anode across one inter-electrode gap,
    so the bath is modeled as n_faces identical cells in parallel, each carrying
    J_avg * plate_area of current.
    
    Parameters:
    J_avg (float): Average applied current density (A/m^2).
    f_pulse (float): Pulsing frequency in Hz.
    D_pulse (float): Duty cycle.
    temp_C (float): Temperature in Celsius.
    C_in_g_L (float): Copper concentration in g/L.
    geom (dict, optional): Cell geometry settings.
    electrical_params (dict, optional): Contact, plate, solution resistances, kinetics, gap.
        Any keys omitted fall back to the defaults below.
    flow_rate_L_min (float, optional): Electrolyte flow rate in L/min (drives mass transfer).
    
    Returns:
    dict: Heat generation terms (Joule, Overpotentials, Chemical, Total) in Watts,
          plus peak cell voltages, applied current, efficiency and copper production.
    """
    if geom is None:
        geom = {
            'plate_width': 1.0,
            'plate_length': 1.0,
            'd_hole': 0.01,
            'pitch': 0.02,
            'n_faces': 160
        }
    default_params = {
        'kappa_cond': 70.0,
        'd_gap': 0.05, # Anode-cathode gap (m)
        'R_contact': 0.0001,
        'R_plate': 0.00005,
        'R_peripheral': 0.00002,
        'f_contact': 0.5,
        'E_eq': 0.89,
        'E_tn': 1.15,
        'I0_c': 10.0,  # Cathode exchange current (A) for 1 m^2 plate area
        'I0_a': 5.0,   # Anode exchange current (A) for 1 m^2 plate area
        'beta_c': 19.1,
        'beta_a': 19.1,
        # Hydrogen evolution side reaction on copper (assumed literature-typical values; calibrate
        # against measured cell voltage)
        'E_eq_H2': 1.23,  # Equilibrium voltage of water splitting (V)
        'E_tn_H2': 1.48,  # Thermoneutral voltage of water splitting (V)
        'J0_H2': 0.01,    # HER exchange current density on Cu (A/m^2)
        'b_H2': 0.12      # HER Tafel slope (V/decade)
    }
    electrical_params = {**default_params, **(electrical_params or {})}
        
    plate_width = geom['plate_width']
    plate_length = geom['plate_length']
    n_faces = geom['n_faces']
    plate_area = plate_width * plate_length
    
    # Current carried by a single anode-cathode cell (A)
    I_cell_avg = J_avg * plate_area
    
    # Calculate Solution resistance of a single cell gap
    R_sol = electrical_params['d_gap'] / (electrical_params['kappa_cond'] * plate_area)
    
    R_extra_voltage = electrical_params['R_plate'] + electrical_params['R_contact'] + electrical_params['R_peripheral']
    R_extra_heating = electrical_params['R_plate'] + electrical_params['f_contact'] * electrical_params['R_contact']
    
    # Get pulsed metrics
    metrics = calculate_pulsed_ew_metrics(J_avg, f_pulse, D_pulse, flow_rate_L_min, temp_C, C_in_g_L, geom)
    
    # Activation kinetics exchange current density (A/m^2)
    J0_c = electrical_params['I0_c'] / plate_area
    J0_a = electrical_params['I0_a'] / plate_area
    
    beta_c = electrical_params['beta_c']
    beta_a = electrical_params['beta_a']
    
    # Effective duty cycle (continuous DC is D = 1)
    D_eff = 1.0 if (f_pulse == 0.0 or D_pulse >= 1.0) else D_pulse
    
    # Calculate average overpotentials during 'on' time
    J_peak = J_avg / D_eff
    I_cell_peak = I_cell_avg / D_eff
    eta_c_on = (1.0 / beta_c) * np.arcsinh(J_peak / (2.0 * J0_c))
    eta_a_on = (1.0 / beta_a) * np.arcsinh(J_peak / (2.0 * J0_a))
    eta_conc_on = 0.5 * (metrics['eta_conc_jet'] + metrics['eta_conc_dead'])  # average conc overpotential
    
    # Joule heating per cell under pulsing: average power is D * I_peak^2 * R, summed over all cells
    Q_joule_sol = n_faces * (D_eff * I_cell_peak**2 * R_sol)
    Q_joule_extra = n_faces * (D_eff * I_cell_peak**2 * R_extra_heating)
    Q_joule = Q_joule_sol + Q_joule_extra
    
    # Split the applied current into copper deposition and hydrogen evolution (current above the
    # mass-transfer limit evolves hydrogen at the cathode)
    I_app_total = n_faces * I_cell_avg
    I_cu = min(metrics['I_actual'], I_app_total)
    I_h2 = I_app_total - I_cu
    
    # Interface voltage during the 'on' pulse. The cathode is one equipotential metal, so the cell
    # runs at whichever reaction path needs the higher voltage.
    V_if_cu = electrical_params['E_eq'] + eta_a_on + eta_c_on + eta_conc_on
    J_h2_peak = I_h2 / (n_faces * plate_area) / D_eff
    if J_h2_peak > electrical_params['J0_H2']:
        eta_h2_on = electrical_params['b_H2'] * math.log10(J_h2_peak / electrical_params['J0_H2'])
        V_if_h2 = electrical_params['E_eq_H2'] + eta_a_on + eta_h2_on
    else:
        eta_h2_on = 0.0
        V_if_h2 = 0.0
    V_if = max(V_if_cu, V_if_h2)
    
    # Electrochemical heat = interface power minus the enthalpy stored in each reaction product.
    # Split into irreversible (overpotential) and reversible (reaction entropy) parts.
    Q_over = I_app_total * V_if - I_cu * electrical_params['E_eq'] - I_h2 * electrical_params['E_eq_H2']
    Q_chem = (I_cu * (electrical_params['E_eq'] - electrical_params['E_tn'])
              + I_h2 * (electrical_params['E_eq_H2'] - electrical_params['E_tn_H2']))
    Q_total = Q_joule + Q_over + Q_chem
    
    # Peak cell voltage during the 'on' pulse
    V_cell_peak_internal = V_if + I_cell_peak * R_sol
    V_cell_peak_total = V_cell_peak_internal + I_cell_peak * R_extra_voltage
        
    return {
        'Q_joule_sol_W': Q_joule_sol,
        'Q_joule_extra_W': Q_joule_extra,
        'Q_overpotential_W': Q_over,
        'Q_chemical_W': Q_chem,
        'Q_total_W': Q_total,
        'V_cell_peak_internal_V': V_cell_peak_internal,
        'V_cell_peak_total_V': V_cell_peak_total,
        'I_applied_A': I_app_total,
        'I_copper_A': I_cu,
        'I_hydrogen_A': I_h2,
        'eta_H2_V': eta_h2_on,
        'J_lim_jet': metrics['J_lim_jet'],
        'J_lim_dead': metrics['J_lim_dead'],
        'efficiency_percent': metrics['efficiency_percent'],
        'copper_g_min': metrics['copper_g_min']
    }
