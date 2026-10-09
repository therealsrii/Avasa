
import numpy as np
from scipy.optimize import differential_evolution
import pulsed_ew_model

# Constraints
MIN_EFFICIENCY = 80.0     # %
MAX_TEMPERATURE_SS = 50.0  # °C (safe limit for PVC/softening & fumes)

# Fixed parameters
n_faces = 40
N_sets = n_faces / 2.0
rho_sol = 1140.0
C_p = 3500.0
U = 10.0
T_amb = 25.0
d_gap = 0.05

def objective(x):
    """
    x[0] = J_avg (50 to 200 A/m^2)
    x[1] = f_pulse (10 to 60 Hz)
    x[2] = D_pulse (0.05 to 0.50)  <-- CAPPED at 50%
    x[3] = flow_rate (175 to 350 L/min)
    x[4] = temp_inlet (30 to 45 °C)
    x[5] = plate_dim (0.4 to 1.0 m)
    """
    J_avg = float(x[0])
    f_pulse = float(x[1])
    D_pulse = float(x[2])
    flow_rate = float(x[3])
    temp_inlet = float(x[4])
    plate_dim = float(x[5])
    
    geom = {
        'plate_width': plate_dim,
        'plate_length': plate_dim,
        'd_hole': 0.01,
        'pitch': 0.02,
        'n_faces': n_faces
    }
    
    # Calculate electrowinning metrics
    metrics = pulsed_ew_model.calculate_pulsed_ew_metrics(
        J_avg=J_avg,
        f_pulse=f_pulse,
        D_pulse=D_pulse,
        flow_rate_L_min=flow_rate,
        temp_C=temp_inlet,
        C_in_g_L=3.0,
        geom=geom
    )
    
    efficiency = metrics['efficiency_percent']
    copper_g_min = metrics['copper_g_min']
    
    # Calculate heat generation
    # Electrical params matching simulation
    electrical_params = {
        'kappa_cond': 70.0,
        'R_contact': 0.0001,
        'R_plate': 0.00005,
        'R_peripheral': 0.00002,
        'f_contact': 0.5,
        'E_eq': 0.89,
        'E_tn': 1.15,
        'I0_c': 10.0,
        'I0_a': 5.0,
        'beta_c': 19.1,
        'beta_a': 19.1
    }
    
    heat_terms = pulsed_ew_model.calculate_heat_generation(
        J_avg=J_avg,
        f_pulse=f_pulse,
        D_pulse=D_pulse,
        temp_C=temp_inlet,
        C_in_g_L=3.0,
        geom=geom,
        electrical_params=electrical_params,
        flow_rate_L_min=flow_rate
    )
    Q_gen = heat_terms['Q_total_W']
    
    # Calculate steady-state temperature under flow
    flow_m3_s = flow_rate / (1000.0 * 60.0)
    m_dot = rho_sol * flow_m3_s
    m_dot_Cp = m_dot * C_p
    
    A_loss = 2.0 * (plate_dim * plate_dim) + (4.0 * N_sets * plate_dim * d_gap)
    UA = U * A_loss
    
    T_ss = (Q_gen + UA * T_amb + m_dot_Cp * temp_inlet) / (UA + m_dot_Cp)
    
    # Minimize negative copper production
    obj_val = -copper_g_min
    
    # Penalty 1: Efficiency constraint
    if efficiency < MIN_EFFICIENCY:
        obj_val += 1e5 * (MIN_EFFICIENCY - efficiency)**2
        
    # Penalty 2: Temperature constraint
    if T_ss > MAX_TEMPERATURE_SS:
        obj_val += 1e5 * (T_ss - MAX_TEMPERATURE_SS)**2
        
    return obj_val

def run_optimization():
    # Bounds: J_avg, f_pulse, D_pulse (capped at 0.5), flow_rate, temp_inlet, plate_dim
    bounds = [
        (50.0, 200.0),    # J_avg (A/m^2)
        (10.0, 60.0),     # f_pulse (Hz)
        (0.05, 0.50),     # D_pulse (Capped at 50%)
        (175.0, 350.0),   # flow_rate (L/min)
        (30.0, 45.0),     # temp_inlet (°C)
        (0.4, 1.0)        # plate_dim (m)
    ]
    
    print("Running Safe Parameters Optimization...")
    res = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=150,
        popsize=20,
        seed=42,
        disp=False
    )
    
    # Optimal results
    opt_J_avg, opt_f_pulse, opt_D_pulse, opt_flow_rate, opt_temp_inlet, opt_plate_dim = res.x
    
    # Calculate final metrics at optimum
    geom = {
        'plate_width': opt_plate_dim,
        'plate_length': opt_plate_dim,
        'd_hole': 0.01,
        'pitch': 0.02,
        'n_faces': n_faces
    }
    
    metrics = pulsed_ew_model.calculate_pulsed_ew_metrics(
        J_avg=opt_J_avg,
        f_pulse=opt_f_pulse,
        D_pulse=opt_D_pulse,
        flow_rate_L_min=opt_flow_rate,
        temp_C=opt_temp_inlet,
        C_in_g_L=3.0,
        geom=geom
    )
    
    heat_terms = pulsed_ew_model.calculate_heat_generation(
        J_avg=opt_J_avg,
        f_pulse=opt_f_pulse,
        D_pulse=opt_D_pulse,
        temp_C=opt_temp_inlet,
        C_in_g_L=3.0,
        geom=geom,
        flow_rate_L_min=opt_flow_rate
    )
    Q_gen = heat_terms['Q_total_W']
    
    flow_m3_s = opt_flow_rate / (1000.0 * 60.0)
    m_dot = rho_sol * flow_m3_s
    m_dot_Cp = m_dot * C_p
    A_loss = 2.0 * (opt_plate_dim * opt_plate_dim) + (4.0 * N_sets * opt_plate_dim * d_gap)
    UA = U * A_loss
    T_ss = (Q_gen + UA * T_amb + m_dot_Cp * opt_temp_inlet) / (UA + m_dot_Cp)
    
    copper_g_min = metrics['copper_g_min']
    copper_kg_hr = (copper_g_min * 60.0) / 1000.0
    annual_prod = copper_kg_hr * 365.25 * 24.0 * 0.80
    
    print("\nSAFE OPTIMIZATION RESULTS:")
    print(f"Optimal J_avg:            {opt_J_avg:.4f} A/m^2")
    print(f"Optimal Pulsing Freq:     {opt_f_pulse:.4f} Hz")
    print(f"Optimal Duty Cycle:       {opt_D_pulse:.4f}")
    print(f"Optimal Flow Rate:        {opt_flow_rate:.4f} L/min")
    print(f"Optimal Inlet Temp:       {opt_temp_inlet:.4f} °C")
    print(f"Optimal Plate Size:       {opt_plate_dim*1000:.1f} mm x {opt_plate_dim*1000:.1f} mm")
    print(f"--------------------------------------------------")
    print(f"Current Efficiency:       {metrics['efficiency_percent']:.4f}%")
    print(f"Steady-State Temperature: {T_ss:.2f} °C")
    print(f"Temperature Rise (dT):    {T_ss - opt_temp_inlet:.2f} °C")
    print(f"Heat Generation (Q_gen):  {Q_gen/1000.0:.2f} kW")
    print(f"Copper Production Rate:   {copper_kg_hr:.6f} kg/hour ({copper_g_min:.4f} g/min)")
    print(f"Annual Copper Production: {annual_prod:.2f} kg/year")

if __name__ == '__main__':
    run_optimization()
