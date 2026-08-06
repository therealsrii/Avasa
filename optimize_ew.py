import sys
import numpy as np
from scipy.optimize import differential_evolution
import pulsed_ew_model

# Target efficiency constraint
MIN_EFFICIENCY = 80.0  # %

def objective(x):
    """
    Objective function for optimization:
    x[0] = J_avg (50 to 200 A/m^2)
    x[1] = f_pulse (10 to 60 Hz)
    x[2] = D_pulse (0.05 to 1.0)
    x[3] = flow_rate_L_min (175 to 350 L/min)
    x[4] = temp_C (30 to 45 °C)
    x[5] = plate_dim (0.4 to 1.0 m)
    """
    J_avg = float(x[0])
    f_pulse = float(x[1])
    D_pulse = float(x[2])
    flow_rate = float(x[3])
    temp = float(x[4])
    plate_dim = float(x[5])
    
    geom = {
        'plate_width': plate_dim,
        'plate_length': plate_dim,
        'd_hole': 0.01,
        'pitch': 0.02,
        'n_faces': 40  # 40 sets of plates
    }
    
    metrics = pulsed_ew_model.calculate_pulsed_ew_metrics(
        J_avg=J_avg,
        f_pulse=f_pulse,
        D_pulse=D_pulse,
        flow_rate_L_min=flow_rate,
        temp_C=temp,
        C_in_g_L=3.0,
        geom=geom
    )
    
    efficiency = metrics['efficiency_percent']
    copper_g_min = metrics['copper_g_min']
    
    # Minimize negative copper production (maximize production)
    obj_val = -copper_g_min
    
    # Quadratic penalty for violating the efficiency constraint
    if efficiency < MIN_EFFICIENCY:
        penalty = 1e5 * (MIN_EFFICIENCY - efficiency)**2
        obj_val += penalty
        
    return obj_val

def run_optimization():
    # Bounds for the variables
    bounds = [
        (50.0, 200.0),    # J_avg (A/m^2)
        (10.0, 60.0),     # f_pulse (Hz)
        (0.05, 1.0),      # D_pulse
        (175.0, 350.0),   # flow_rate (L/min)
        (30.0, 45.0),     # temp_C
        (0.4, 1.0)        # plate_dim (m)
    ]
    
    print("Running Differential Evolution Optimization...")
    res = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        disp=False
    )
    
    # Decode optimal results
    opt_J_avg, opt_f_pulse, opt_D_pulse, opt_flow_rate, opt_temp, opt_plate_dim = res.x
    
    # Re-calculate metrics at optimum
    geom = {
        'plate_width': opt_plate_dim,
        'plate_length': opt_plate_dim,
        'd_hole': 0.01,
        'pitch': 0.02,
        'n_faces': 40
    }
    
    metrics = pulsed_ew_model.calculate_pulsed_ew_metrics(
        J_avg=opt_J_avg,
        f_pulse=opt_f_pulse,
        D_pulse=opt_D_pulse,
        flow_rate_L_min=opt_flow_rate,
        temp_C=opt_temp,
        C_in_g_L=3.0,
        geom=geom
    )
    
    copper_g_min = metrics['copper_g_min']
    copper_kg_hr = (copper_g_min * 60.0) / 1000.0
    
    print("\nOPTIMIZATION RESULTS:")
    print(f"Optimal J_avg:            {opt_J_avg:.4f} A/m^2")
    print(f"Optimal Pulsing Freq:     {opt_f_pulse:.4f} Hz")
    print(f"Optimal Duty Cycle:       {opt_D_pulse:.4f}")
    print(f"Optimal Flow Rate:        {opt_flow_rate:.4f} L/min")
    print(f"Optimal Temp:             {opt_temp:.4f} °C")
    print(f"Optimal Plate Size:       {opt_plate_dim*1000:.1f} mm x {opt_plate_dim*1000:.1f} mm")
    print(f"--------------------------------------------------")
    print(f"Current Efficiency:       {metrics['efficiency_percent']:.4f}%")
    print(f"Copper Production Rate:   {copper_kg_hr:.6f} kg/hour ({copper_g_min:.4f} g/min)")
    print(f"Outlet Concentration:     {metrics['C_out_g_L']:.6f} g/L")
    print(f"Applied Average Current:  {metrics['I_applied']:.2f} A")
    print(f"Actual Deposition Current:{metrics['I_actual']:.2f} A")

if __name__ == '__main__':
    run_optimization()
