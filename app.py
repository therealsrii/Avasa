import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
import importlib
import pulsed_ew_model as ewm

# Streamlit reruns this script on every change but keeps imported modules cached, so a
# redeploy that updates the model would otherwise keep running the old version.
importlib.reload(ewm)

# Set page config for a premium wide layout
st.set_page_config(
    page_title="Copper Electrowinning Thermal Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sleek, zinc-colored, premium look
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #fafafa;
        color: #18181b;
        font-family: 'Inter', sans-serif;
    }
    
    /* Force dark text for all body paragraphs, labels, and spans */
    .stApp p, .stApp label, .stApp span, .stApp li, .stApp div {
        color: #18181b !important;
    }
    
    /* Title and headers */
    h1 {
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #09090b !important;
    }
    h2, h3 {
        font-weight: 700;
        color: #18181b !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f4f4f5;
        border-right: 1px solid #e4e4e7;
    }
    section[data-testid="stSidebar"] h2 {
        color: #09090b !important;
        font-size: 1.25rem;
    }
    
    /* Specific overrides for sidebar labels and widget headers */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] span {
        color: #18181b !important;
        font-weight: 600 !important;
    }
    
    /* Radio options list text color */
    div[data-testid="stRadio"] label p {
        color: #27272a !important;
        font-weight: 400 !important;
    }
    
    /* Card design for metrics */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e4e4e7;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #71717a !important;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #09090b !important;
        letter-spacing: -0.03em;
    }
    .metric-subtext {
        font-size: 0.75rem;
        color: #a1a1aa !important;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR PARAMETERS
# ==========================================
st.sidebar.header("Cell Parameters")

# Operating Mode Toggle
mode = st.sidebar.radio(
    "Pulsed Current Mode",
    ["Constant Average Current", "Constant Peak Current"],
    help="Constant Average: Keeps deposition rate constant but causes high Joule heating at low duty cycles.\nConstant Peak: Keeps peak power constant, meaning average deposition rate drops with duty cycle."
)

# Sliders for physical properties
st.sidebar.subheader("Cell Geometry")
N_sets = st.sidebar.slider("Number of Electrode Sets", min_value=1, max_value=100, value=8, step=1, help="Number of identical anode-cathode pairs operating in the bath, connected in parallel to the rectifier.")
A_plate = st.sidebar.slider("Plate Area (m²)", min_value=0.1, max_value=4.0, value=1.0, step=0.1, help="Area of the anode and cathode plates.")
d_gap_cm = st.sidebar.slider("Plate Gap Distance (cm)", min_value=1.0, max_value=20.0, value=5.0, step=0.5, help="Distance separating the anode and cathode.")
d_gap = d_gap_cm / 100.0 # Convert to meters

# Dynamically calculate volume range to prevent physical impossibility
vol_between_plates_L = int(N_sets * A_plate * d_gap * 1000.0)
min_vol = max(10, vol_between_plates_L + 10)
max_vol = max(500, vol_between_plates_L + 3000)
default_vol = max(50, vol_between_plates_L + 500)
step_vol = 50 if max_vol > 1000 else 10

vol_sol_L = st.sidebar.slider("Solution Volume (L)", min_value=min_vol, max_value=max_vol, value=default_vol, step=step_vol, help="Total volume of the H2SO4 copper electrolyte bath.")
vol_sol = vol_sol_L / 1000.0 # Convert to m^3

st.sidebar.subheader("Electrical & Chemical")
I_target = st.sidebar.slider("Current per Electrode Set (A)", min_value=10, max_value=1000, value=350, step=10, help="Target current applied to each anode-cathode set (default is 350 A, corresponding to 350 A/m² on a 1m² active plate area).")
kappa_cond = st.sidebar.slider("Electrolyte Conductivity (S/m)", min_value=10, max_value=120, value=70, step=5, help="Electrical conductivity of 20% H2SO4 with copper. Default is ~70 S/m.")
C_cu_g_L = st.sidebar.slider("Copper Concentration (g/L)", min_value=0.1, max_value=50.0, value=1.0, step=0.1, help="Bulk Cu²⁺ concentration in the electrolyte. Sets the mass-transfer limiting current.")
temp_C = st.sidebar.slider("Electrolyte Temperature (°C)", min_value=20.0, max_value=60.0, value=30.0, step=1.0, help="Electrolyte (inlet) temperature. Sets viscosity, density and Cu diffusivity.")
flow_rate_L_min = st.sidebar.slider("Electrolyte Flow Rate (L/min)", min_value=10, max_value=500, value=350, step=10, help="Electrolyte flow through the cathode jet holes. Drives mass transfer and, in continuous-flow mode, carries heat out of the cell.")

st.sidebar.subheader("Parasitic Resistance & Contacts")
R_contact_mohm = st.sidebar.slider("Contact Resistance (mΩ)", min_value=0.0, max_value=2.0, value=0.1, step=0.05, help="Electrical contact resistance at the busbar-to-hanger-bar junctions (anode + cathode).")
R_plate_mohm = st.sidebar.slider("Electrode Plate Resistance (mΩ)", min_value=0.0, max_value=0.5, value=0.05, step=0.01, help="Internal resistance of the anode and cathode plates and hanger bars.")
R_peripheral_mohm = st.sidebar.slider("Peripheral/Busbar Resistance (mΩ)", min_value=0.0, max_value=1.0, value=0.02, step=0.01, help="Ohmic resistance of inter-cell connectors, busbars, and cabling.")
f_contact_heat_pct = st.sidebar.slider("Contact Heat Conducted to Bath (%)", min_value=0.0, max_value=100.0, value=50.0, step=5.0, help="Percentage of the contact resistance heat that conducts into the electrolyte bath via the hanger bars.")

st.sidebar.subheader("Pulsing Parameters")
f_pulse = st.sidebar.slider("Pulsing Frequency (Hz)", min_value=1.0, max_value=1000.0, value=30.0, step=1.0, help="Operating pulsing frequency of the current.")
D_pulse = st.sidebar.slider("Duty Cycle (D)", min_value=0.05, max_value=1.0, value=0.75, step=0.05, help="Operating duty cycle of the pulses.")

st.sidebar.subheader("Production Schedule")
hours_per_day = st.sidebar.slider("Operating Hours per Day", min_value=1, max_value=24, value=12, step=1, help="Hours of electrowinning operation per day.")
days_per_year = st.sidebar.slider("Operating Days per Year", min_value=1, max_value=365, value=365, step=1, help="Days of operation per year. The yearly yield assumes the copper concentration is held constant (e.g. continuously replenished by solvent extraction).")

st.sidebar.subheader("Rectifier")
I_rect_max = st.sidebar.slider("Rectifier Max Current (A)", min_value=500, max_value=10000, value=4500, step=100, help="Rated output current of the rectifier. All electrode sets are fed in parallel, so it must supply the peak current of every set at once.")
V_rect_max = st.sidebar.slider("Rectifier Max Voltage (V)", min_value=1.0, max_value=20.0, value=6.0, step=0.5, help="Rated output voltage of the rectifier. With the sets in parallel, it must supply the peak cell voltage.")

st.sidebar.subheader("Heat Transfer")
thermal_mode = st.sidebar.radio(
    "Thermal Model",
    ["Closed Bath (Transient)", "Continuous Flow (Steady State)"],
    help="Closed Bath: no fresh feed, the bath heats up over the operation duration.\nContinuous Flow: fresh electrolyte enters at the electrolyte temperature and carries heat out; reports the steady-state rise."
)
U_coeff = st.sidebar.slider("Heat Loss Coefficient U (W/m²K)", min_value=0.0, max_value=50.0, value=10.0, step=1.0, help="Heat transfer coefficient to the surrounding air. 0 = Adiabatic cell.")
if thermal_mode == "Closed Bath (Transient)":
    t_run_min = st.sidebar.slider("Operation Duration (min)", min_value=5, max_value=120, value=20, step=5, help="Duration of electrowinning operation.")
else:
    t_run_min = None

# Hidden/Static parameters
rho_sol = 1140.0 # Density (kg/m^3)
m_sol = rho_sol * vol_sol  # Mass of solution (kg)
C_p = 3500.0     # Heat capacity of 20% H2SO4 (J/kg/K)
T_amb = 25.0     # Ambient temperature (°C)

# Cell geometry passed to the core model. Each electrode set is one anode-cathode
# gap, i.e. one active cathode face.
geom = {
    'plate_width': np.sqrt(A_plate),
    'plate_length': np.sqrt(A_plate),
    'd_hole': 0.01,
    'pitch': 0.02,
    'n_faces': N_sets
}

# Electrical and kinetic parameters passed to the core model
electrical_params = {
    'kappa_cond': float(kappa_cond),
    'd_gap': d_gap,
    'R_contact': R_contact_mohm / 1000.0,
    'R_plate': R_plate_mohm / 1000.0,
    'R_peripheral': R_peripheral_mohm / 1000.0,
    'f_contact': f_contact_heat_pct / 100.0,
    'E_eq': 0.89,     # Equilibrium cell potential (V)
    'E_tn': 1.15,     # Thermoneutral voltage (V)
    'I0_c': 10.0,     # Cathode exchange current (A)
    'I0_a': 5.0,      # Anode exchange current (A)
    'beta_c': 19.1,   # Cathode charge transfer coeff (V^-1)
    'beta_a': 19.1    # Anode charge transfer coeff (V^-1)
}

# ==========================================
# SOLVER IMPLEMENTATION (physics from pulsed_ew_model.py)
# ==========================================

def calculate_cell_metrics(f, D, mode_select, N_sets=1):
    """
    Calculates cell voltages, powers, efficiency and temperature rise for the whole bath (N_sets)
    using the core pulsed electrowinning model.
    """
    if mode_select == "Constant Average Current":
        I_avg = float(I_target)
    else:
        I_avg = float(I_target) * D
    J_avg = I_avg / A_plate
    
    heat = ewm.calculate_heat_generation(
        J_avg, f, D, temp_C, C_cu_g_L,
        geom=geom, electrical_params=electrical_params, flow_rate_L_min=flow_rate_L_min
    )
    Q_gen = heat['Q_total_W']
    
    # Temperature rise calculations with convective heat losses
    # Total heat loss area scales with number of cells in the tank
    A_loss = 2.0 * A_plate + (4.0 * N_sets * np.sqrt(A_plate) * d_gap)
    UA = U_coeff * A_loss
    
    if thermal_mode == "Closed Bath (Transient)":
        mC = m_sol * C_p
        t_run = t_run_min * 60.0
        if UA == 0:
            dT = (Q_gen * t_run) / mC
        else:
            dT = (Q_gen / UA) * (1.0 - np.exp(-(UA / mC) * t_run))
    else:
        # Steady state with fresh electrolyte entering at temp_C; rise is relative to the inlet
        m_dot_Cp = rho_sol * (flow_rate_L_min / 60000.0) * C_p
        T_ss = (Q_gen + UA * T_amb + m_dot_Cp * temp_C) / (UA + m_dot_Cp)
        dT = T_ss - temp_C
        
    return heat, dT

# ==========================================
# MAIN PAGE LAYOUT
# ==========================================

st.title("Copper Electrowinning Thermal Simulator")
st.markdown("""
This interactive simulation models the thermal dynamics of a pulsed-current copper electrowinning cell system.
You can adjust the parameters in the sidebar to see how cell geometry, electrical properties, and heat loss coefficients affect the temperature rise.
""")

# Calculate current active operating point metrics using the sidebar pulsing sliders
heat, dT_final = calculate_cell_metrics(f_pulse, D_pulse, mode, N_sets)
V_on_int = heat['V_cell_peak_internal_V']
V_on_total = heat['V_cell_peak_total_V']
P_joule_sol = heat['Q_joule_sol_W']
P_joule_extra = heat['Q_joule_extra_W']
P_over = heat['Q_overpotential_W']
P_chem = heat['Q_chemical_W']
P_gen = heat['Q_total_W']
efficiency = heat['efficiency_percent']
copper_kg_hr = heat['copper_g_min'] * 60.0 / 1000.0
hours_per_year = hours_per_day * days_per_year
copper_kg_yr = copper_kg_hr * hours_per_year

if thermal_mode == "Closed Bath (Transient)":
    dT_caption = f"Temperature Rise after <b>{t_run_min} min</b> of operation"
    dT_axis = f"Temperature Rise after {t_run_min} min"
else:
    dT_caption = f"Steady-State Rise above {temp_C:.0f} °C inlet at <b>{flow_rate_L_min} L/min</b>"
    dT_axis = "Steady-State Temperature Rise"
P_joule = P_joule_sol + P_joule_extra

# Callout box for the whole bath forecast (N_sets)
st.markdown(f"""
<div class="metric-card" style="background-color: #fef2f2; border: 1px solid #fee2e2; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
    <div style="font-size: 0.875rem; font-weight: 700; color: #ef4444; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Whole Bath Thermal Forecast ({N_sets} Electrode Sets)</div>
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <div style="font-size: 2.25rem; font-weight: 800; color: #b91c1c; letter-spacing: -0.03em;">+{dT_final:.2f} °C</div>
            <div style="font-size: 0.85rem; color: #4b5563; margin-top: 2px;">{dT_caption} (at {f_pulse:.1f} Hz, {D_pulse*100:.0f}% D)</div>
            <div style="font-size: 0.85rem; color: #4b5563; margin-top: 6px;">Current to Copper <b>{efficiency:.1f}%</b> (rest evolves H₂) · Cu Deposition <b>{copper_kg_hr:.3f} kg/h</b></div>
        </div>
        <div style="text-align: right; min-width: 200px;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #09090b;">{P_gen/1000.0:.2f} kW</div>
            <div style="font-size: 0.85rem; color: #4b5563; margin-top: 2px;">Total Heat Generation Rate</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Callout box for the yearly copper production forecast
st.markdown(f"""
<div class="metric-card" style="background-color: #f0fdf4; border: 1px solid #dcfce7; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
    <div style="font-size: 0.875rem; font-weight: 700; color: #15803d; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Yearly Copper Yield ({hours_per_day} h/day × {days_per_year} days = {hours_per_year:,} h)</div>
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <div style="font-size: 2.25rem; font-weight: 800; color: #166534; letter-spacing: -0.03em;">{copper_kg_yr:,.0f} kg/yr</div>
            <div style="font-size: 0.85rem; color: #4b5563; margin-top: 2px;">{copper_kg_hr:.3f} kg/h at {efficiency:.1f}% current efficiency ({C_cu_g_L:.1f} g/L Cu held constant)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Rectifier capacity check (electrode sets are in parallel: currents add, voltage is one cell)
I_set_peak = float(I_target) / D_pulse if mode == "Constant Average Current" else float(I_target)
I_rect_peak = N_sets * I_set_peak
I_rect_avg = N_sets * (float(I_target) if mode == "Constant Average Current" else float(I_target) * D_pulse)
rect_problems = []
if I_rect_peak > I_rect_max:
    rect_problems.append(f"peak current **{I_rect_peak:,.0f} A** exceeds the **{I_rect_max:,.0f} A** rating")
if V_on_total > V_rect_max:
    rect_problems.append(f"peak voltage **{V_on_total:.2f} V** exceeds the **{V_rect_max:.1f} V** rating")
if rect_problems:
    msg = "**Rectifier overloaded:** " + " and ".join(rect_problems) + "."
    if mode == "Constant Average Current" and I_rect_avg <= I_rect_max:
        msg += f" At this average current the duty cycle must be at least **{np.ceil(100 * I_rect_avg / I_rect_max) / 100:.2f}**."
    st.error(msg)
else:
    st.success(f"**Rectifier OK:** peak demand {I_rect_peak:,.0f} A ({100 * I_rect_peak / I_rect_max:.0f}% of {I_rect_max:,.0f} A) at {V_on_total:.2f} V ({100 * V_on_total / V_rect_max:.0f}% of {V_rect_max:.1f} V).")

# Deposit regime: average current density relative to the mass-transfer limit in each zone.
# Above the limit copper grows as loose dendritic powder; below it, as a compact adherent layer.
J_avg_op = (float(I_target) if mode == "Constant Average Current" else float(I_target) * D_pulse) / A_plate
ratio_jet = J_avg_op / heat['J_lim_jet']
ratio_dead = J_avg_op / heat['J_lim_dead']

def deposit_regime(ratio):
    if ratio < 1.0:
        return "Compact (sticks to plate)", "#b91c1c"
    if ratio < 1.5:
        return "Borderline", "#b45309"
    return "Powder", "#15803d"

regime_jet, color_jet = deposit_regime(ratio_jet)
regime_dead, color_dead = deposit_regime(ratio_dead)

# Gas evolution at 25 °C, 1 atm (24.45 L/mol). Hydrogen from current above the copper limit at the
# cathode; oxygen from all current at the anode.
V_molar_L = 24.45
H2_L_min = heat['I_hydrogen_A'] / (2.0 * ewm.F) * V_molar_L * 60.0
O2_L_min = heat['I_applied_A'] / (4.0 * ewm.F) * V_molar_L * 60.0
# Extraction air to dilute H2 to 1% by volume (25% of its 4% lower flammability limit)
vent_m3_h = H2_L_min * 60.0 / 1000.0 / 0.01

reg_col, gas_col = st.columns(2)
with reg_col:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Deposit Regime (J / J<sub>lim</sub>)</div>
        <div style="display: flex; justify-content: space-between; margin-top: 6px;">
            <div>
                <div style="font-size: 0.8rem; color: #71717a;">Jet zones</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{ratio_jet:.1f}×</div>
                <div style="font-size: 0.85rem;"><strong style="color: {color_jet};">{regime_jet}</strong></div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; color: #71717a;">Dead zones</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{ratio_dead:.1f}×</div>
                <div style="font-size: 0.85rem;"><strong style="color: {color_dead};">{regime_dead}</strong></div>
            </div>
        </div>
        <div class="metric-subtext">J = {J_avg_op:.0f} A/m² vs limit {heat['J_lim_jet']:.0f} (jet) / {heat['J_lim_dead']:.0f} (dead) A/m². Powder above 1.5× is a rule of thumb; confirm by test.</div>
    </div>
    """, unsafe_allow_html=True)
with gas_col:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Gas Evolution</div>
        <div style="display: flex; justify-content: space-between; margin-top: 6px;">
            <div>
                <div style="font-size: 0.8rem; color: #71717a;">H₂ (cathode)</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{H2_L_min:.1f} L/min</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; color: #71717a;">O₂ (anode)</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{O2_L_min:.1f} L/min</div>
            </div>
        </div>
        <div class="metric-subtext">Keeping H₂ below 1% in the headspace needs roughly <b>{vent_m3_h:,.0f} m³/h</b> of extraction air (indicative only; design ventilation to the applicable safety standards).</div>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Peak Cell Voltage</div>
        <div class="metric-value">{V_on_total:.2f} V</div>
        <div class="metric-subtext">Rectifier peak ({V_on_int:.2f} V internal)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">System Joule Heat</div>
        <div class="metric-value">{P_joule/1000.0:.2f} kW</div>
        <div class="metric-subtext">Sol: {P_joule_sol/1000.0:.2f} kW | Parasitic: {P_joule_extra/1000.0:.2f} kW</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">System Overpotential</div>
        <div class="metric-value">{P_over/1000.0:.2f} kW</div>
        <div class="metric-subtext">Activation & concentration heat</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">System Reaction Heat</div>
        <div class="metric-value">{P_chem/1000.0:.2f} kW</div>
        <div class="metric-subtext">Endothermic net absorption</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total System Heat</div>
        <div class="metric-value" style="color: #ef4444;">{P_gen/1000.0:.2f} kW</div>
        <div class="metric-subtext">Net heat generation rate</div>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["3D Thermal Surface", "Model Physics & Equations"])

with tab1:
    st.subheader(f"3D Thermal Surface Plot ({mode})")
    
    # Generate Grid
    freqs = np.logspace(0.0, 3.0, 25)  # 1 to 1000 Hz
    duties = np.linspace(0.05, 1.0, 25)
    F_grid, D_grid = np.meshgrid(freqs, duties)
    
    # Calculate Z (Temperature Rise)
    Z_grid = np.zeros_like(F_grid)
    for i in range(len(duties)):
        for j in range(len(freqs)):
            _, dT = calculate_cell_metrics(F_grid[i, j], D_grid[i, j], mode, N_sets)
            Z_grid[i, j] = dT
            
    # Calculate DC for plotting at f=0 boundary if desired
    # The surface spans the 1-1000 Hz slider range on a log axis.
    
    # Create Plotly 3D Surface
    colorscale = 'Magma' if mode == "Constant Average Current" else 'Viridis'
    
    fig = go.Figure(data=[go.Surface(
        x=np.log10(F_grid),
        customdata=F_grid,
        y=D_grid,
        z=Z_grid,
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text="Temp Rise (°C)", font=dict(color="#09090b", size=12)),
            tickfont=dict(color="#27272a", size=10)
        ),
        hovertemplate=(
            "Frequency: %{customdata:.1f} Hz<br>" +
            "Duty Cycle: %{y:.2f}<br>" +
            "Temp Rise: %{z:.3f} °C<br>" +
            "<extra></extra>"
        )
    )])
    
    # Add a marker point on the surface representing the current operating point selected via the sliders
    fig.add_trace(go.Scatter3d(
        x=[np.log10(f_pulse)],
        customdata=[f_pulse],
        y=[D_pulse],
        z=[dT_final],
        mode='markers',
        marker=dict(
            size=9,
            color='#ef4444',
            symbol='circle',
            line=dict(color='#ffffff', width=2)
        ),
        name='Operating Point',
        hovertemplate=(
            "Current Operating Point:<br>" +
            "Frequency: %{customdata:.1f} Hz<br>" +
            "Duty Cycle: %{y:.2f}<br>" +
            "Temp Rise: %{z:.3f} °C<br>" +
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Whole Bath {dT_axis} ({N_sets} Sets, Current: {I_target} A, Volume: {vol_sol_L} L, Mode: {mode})",
            font=dict(color="#09090b", size=14)
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text="Pulsing Frequency (Hz)", font=dict(color="#09090b", size=12)),
                tickvals=[0, 1, np.log10(60), 2, 3],
                ticktext=["1", "10", "60", "100", "1000"],
                tickfont=dict(color="#27272a", size=10),
                gridcolor="rgb(200, 200, 200)",
                showbackground=True,
                backgroundcolor="rgb(244, 244, 245)",
                zerolinecolor="rgb(161, 161, 170)"
            ),
            yaxis=dict(
                title=dict(text="Duty Cycle (D)", font=dict(color="#09090b", size=12)),
                tickfont=dict(color="#27272a", size=10),
                gridcolor="rgb(200, 200, 200)",
                showbackground=True,
                backgroundcolor="rgb(244, 244, 245)",
                zerolinecolor="rgb(161, 161, 170)"
            ),
            zaxis=dict(
                title=dict(text="Temperature Rise (°C)", font=dict(color="#09090b", size=12)),
                tickfont=dict(color="#27272a", size=10),
                gridcolor="rgb(200, 200, 200)",
                showbackground=True,
                backgroundcolor="rgb(244, 244, 245)",
                zerolinecolor="rgb(161, 161, 170)"
            ),
            bgcolor="rgb(255, 255, 255)"
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        width=1000,
        height=650,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Physics observations based on current selection
    st.markdown("### Key Model Insights")
    mt = ewm.mass_transfer_coefficients(flow_rate_L_min, temp_C, geom)
    if mode == "Constant Average Current":
        st.markdown(f"""
        - **Joule Heating Blowup**: Because the average current is fixed at **{I_target} A**, decreasing the duty cycle forces the peak current to rise as $I_{{peak}} = I_{{avg}}/D$. Ohmic heating increases quadratically with peak current ($I_{{peak}}^2 R$), resulting in a net Joule heating scaling of $1/D$. At $D = 0.05$, peak current is **{I_target/0.05:.0f} A**, causing a very steep temperature rise.
        - **Frequency Flatness**: The surface is nearly flat along the frequency axis. This occurs because the electrical double-layer charges in microseconds ($\\tau \\approx 17\\ \\mu\\text{{s}}$) and the concentration diffusion layer relaxes slowly ($\\tau_{{jet}} \\approx {mt['tau_jet']:.1f}\\ \\text{{s}}$, $\\tau_{{dead}} \\approx {mt['tau_dead']:.0f}\\ \\text{{s}}$ at {flow_rate_L_min} L/min). Both systems settle into steady-state cycles quickly, making frequency thermally neutral in the $1-1000\\ \\text{{Hz}}$ range.
        """)
    else:
        st.markdown(f"""
        - **Linear Heat Decrease**: Since the peak current is fixed at **{I_target} A**, reducing the duty cycle decreases the average current linearly ($I_{{avg}} = I_{{peak}} \\cdot D$). Both Joule heating ($D \\cdot I_{{peak}}^2 R$) and overpotential power drop linearly with $D$, causing the temperature rise to drop to nearly 0°C at very low duty cycles.
        - **Production Trade-off**: While a lower duty cycle reduces solution heating, it also reduces the copper deposition rate proportionally due to the lower average current.
        """)
    if ratio_jet >= 1.0:
        mt_text = (f"The applied **{J_avg_op:.0f} A/m²** is above the limit in both zones, so copper grows as loose powder "
                   f"and the remaining current evolves hydrogen; **{efficiency:.1f}%** of the current deposits copper. "
                   "Low current efficiency is expected when making powder.")
    elif ratio_dead >= 1.0:
        mt_text = (f"The applied **{J_avg_op:.0f} A/m²** is above the limit only in the dead zones. Expect a **mixed deposit**: "
                   "powder in the dead zones, but compact copper that sticks to the plate around the jet holes "
                   "(and may narrow them). Lower the copper concentration or flow, or raise the current density, to get powder everywhere.")
    else:
        mt_text = (f"The applied **{J_avg_op:.0f} A/m²** is below the limit in both zones, so copper deposits as a "
                   f"**compact layer that sticks to the plate**, not powder (efficiency **{efficiency:.1f}%**).")
    st.markdown(f"""
        - **Mass-Transfer Limit and Deposit Type**: At {C_cu_g_L:.1f} g/L Cu and {flow_rate_L_min} L/min, copper can reach the cathode at most at **{heat['J_lim_jet']:.0f} A/m²** in the jet zones and **{heat['J_lim_dead']:.0f} A/m²** in the dead zones. {mt_text}
        - **What Sets the Yield**: Above the limit, extra current makes hydrogen, not copper. Copper yield is set by cathode area times the limiting current, so it rises with flow rate, temperature, copper concentration and plate area. Spreading the same rectifier current over more plates (lower A/m², still above the limit) gives more powder.
        - **Why Frequency Does Not Help Yield**: Averaged over time, copper can reach the cathode no faster than the limiting current allows, no matter how the current is pulsed. Pulse frequency may still affect powder particle size and shape, which this model does not predict.
        """)

with tab2:
    st.subheader("Mathematical and Physical Framework")
    st.markdown(r"""
    This simulator models the coupling of electrochemical reaction kinetics, mass transport, and thermal dynamics in a pulsed-current electrowinning cell system. All physics is computed by the core model in `pulsed_ew_model.py`, the same module used by the optimization scripts.

    ### 1. Cell Electrical Model and Overpotentials
    The transient voltage required to drive the electrowinning cell is modeled as:
    
    $$V_{cell, total}(t) = V_{cell, internal}(t) + I(t) \cdot (R_{plate} + R_{contact} + R_{peripheral})$$
    
    where:
    
    $$V_{cell, internal}(t) = E_{eq} + \eta_a(t) + \eta_c(t) + \eta_{conc}(t) + I(t) R_{sol}$$

    Where the individual components represent:
    
    *   **Thermodynamic Equilibrium Potential ($E_{eq}$)**: The thermodynamic potential required for the net copper electrowinning reaction:
        $$\text{Cu}^{2+} + \text{H}_2\text{O} \rightarrow \text{Cu} + \frac{1}{2}\text{O}_2 + 2\text{H}^+$$
        
    *   **Activation Overpotentials ($\eta_a, \eta_c$)**: The kinetic barriers for charge transfer at the anode and cathode surfaces. These are governed by Butler-Volmer kinetics coupled with double-layer capacitance charging:
        $$C_c \frac{d\eta_c}{dt} = I(t) - 2 I_{0,c}\sinh(\beta_c \eta_c)$$
        $$C_a \frac{d\eta_a}{dt} = I(t) - 2 I_{0,a}\sinh(\beta_a \eta_a)$$
        
    *   **Concentration Overpotential ($\eta_{conc}$)**: The voltage loss due to copper ion depletion at the cathode surface. It is governed by the Nernst equation:
        $$\eta_{conc}(t) = -\frac{R_{gas} T}{2F} \ln\left( \frac{C_s(t)}{C_b} \right)$$
        Where the cathode surface concentration $C_s(t)$ is driven by the electrochemical reaction rate and replenished by boundary-layer diffusion:
        $$\frac{dC_s}{dt} = \frac{k(C_b - C_s(t))}{\delta} - \frac{J(t)}{2 F \delta}$$
        The cathode is split into **jet zones** (electrolyte jets through the plate holes) and **dead zones**. The jet mass transfer coefficient comes from a Sherwood correlation $Sh = Re^{0.5} Sc^{0.33}$ using the flow rate and temperature-dependent viscosity and Cu diffusivity; dead zones use $k_{dead} = 0.15\,k_{jet}$. If the peak current density exceeds the limiting current $J_{lim} = 2 F k C_b$, the surface depletes to zero during the pulse and the excess current goes to side reactions, lowering the current efficiency.
        
    *   **Ohmic Resistance ($R_{sol}$)**: The resistance of the bulk sulfuric acid solution separating the electrodes:
        $$R_{sol} = \frac{d_{gap}}{\kappa_{cond} A_{plate}}$$

    *   **Parasitic & Contact Resistance ($R_{contact}, R_{plate}, R_{peripheral}$)**: Ohmic losses from hanger-bar contacts, the bulk electrode plates, and peripheral connections.

    ---

    ### 2. Thermal Energy Balance Model
    The transient temperature rise of the electrolyte bath is governed by the net heat generation rate $Q_{gen}$ and convective heat losses to the ambient environment:

    $$m_{sol} C_p \frac{dT_{sol}}{dt} = Q_{gen, total} - U A_{loss} (T_{sol} - T_{amb})$$

    Key aspects of the thermal model include:
    
    *   **Total Bath Joule Heat**: Ohmic heating occurs in the solution and the plates, plus a fraction of contact resistance heat conducting back through the hanger bars:
        $$Q_{joule, total} = N_{sets} \cdot I^2 \cdot (R_{sol} + R_{plate} + f_{contact} \cdot R_{contact})$$

    *   **Endothermic Reaction Enthalpy Subtraction**: The chemical reaction absorbs energy. The net rate of heat generation is calculated by subtracting the reaction enthalpy, represented as a thermoneutral voltage ($E_{tn} = 1.15\text{ V}$):
        $$Q_{gen, total}(t) = Q_{joule, total}(t) + N_{sets} \cdot I(t) \cdot (\eta_a(t) + \eta_c(t) + \eta_{conc}(t) + E_{eq} - E_{tn})$$
        All applied current crosses the electrode interfaces, so this uses the applied current whether it deposits copper or drives side reactions.
        
    *   **Closed Bath (Transient)**: Integrating the thermal ODE yields the solution temperature rise over the operation duration $t$:
        $$\Delta T(t) = \frac{Q_{gen, avg}}{U A_{loss}} \left( 1 - \exp\left( -\frac{U A_{loss}}{m_{sol} C_p} t \right) \right)$$
        Where $Q_{gen, avg}$ is the average heat generation rate over a pulsing period.

    *   **Continuous Flow (Steady State)**: Fresh electrolyte enters at $T_{in}$ with heat capacity rate $\dot{m} C_p$, giving the steady-state temperature:
        $$T_{ss} = \frac{Q_{gen} + U A_{loss} T_{amb} + \dot{m} C_p T_{in}}{U A_{loss} + \dot{m} C_p}$$
        The reported rise is $T_{ss} - T_{in}$, with $T_{amb} = 25\,^\circ\text{C}$.
        
    *   **Heat Loss Area ($A_{loss}$)**: Calculated based on the geometry of a rectangular tank containing $N_{sets}$ plates separated by gap $d_{gap}$:
        $$A_{loss} = 2 A_{plate} + 4 N_{sets} \sqrt{A_{plate}} d_{gap}$$

    ---

    ### 3. Hydrogen Evolution and Deposit Morphology
    Current above the copper mass-transfer limit evolves hydrogen at the cathode:
    $$I_{H_2} = I_{applied} - I_{Cu}$$
    The cathode is a single equipotential metal, so the cell runs at whichever reaction path needs the higher voltage. The hydrogen path uses Tafel kinetics:
    $$V_{interface} = \max\left(E_{eq} + \eta_a + \eta_c + \eta_{conc},\; E_{eq,H_2} + \eta_a + b_{H_2}\log_{10}\frac{J_{H_2}}{J_{0,H_2}}\right)$$
    Heat generation is the interface power minus the enthalpy stored in each product:
    $$Q_{electrochem} = I_{applied} V_{interface} - I_{Cu} E_{tn} - I_{H_2} E_{tn,H_2}$$
    The ratio of applied to limiting current density, $J/J_{lim}$, indicates deposit morphology: below 1 copper forms a compact adherent layer; above the limit it grows as dendritic powder. The 1.5× powder threshold shown on the dashboard is a rule of thumb that should be confirmed experimentally.

    ---

    ### 4. Model Parameters and Constants Reference Table
    The physical and chemical parameters used to evaluate the electrowinning cell model are listed below:

    | Symbol | Parameter Description | Nominal Value | Unit | Physical Significance |
    | :--- | :--- | :--- | :--- | :--- |
    | $E_{eq}$ | Equilibrium Potential | $0.89$ | $\text{V}$ | Thermodynamic potential for Cu deposition / $\text{O}_2$ evolution |
    | $E_{tn}$ | Thermoneutral Potential | $1.15$ | $\text{V}$ | Cell voltage at which net reaction heat generation is zero |
    | $E_{eq,H_2}, E_{tn,H_2}$ | Water Splitting Potentials | $1.23, 1.48$ | $\text{V}$ | Equilibrium and thermoneutral voltage of the hydrogen side reaction |
    | $J_{0,H_2}$ | HER Exchange Current Density | $0.01$ | $\text{A/m}^2$ | Assumed typical value on copper; calibrate against measured cell voltage |
    | $b_{H_2}$ | HER Tafel Slope | $0.12$ | $\text{V/decade}$ | Assumed typical value; calibrate against measured cell voltage |
    | $C_c, C_a$ | Double-Layer Capacitance | $0.2$ | $\text{F}$ | Cathode & Anode double-layer capacitance |
    | $I_{0,c}$ | Cathode Exchange Current | $10.0$ | $\text{A}$ | Kinematic charge transfer rate at cathode |
    | $I_{0,a}$ | Anode Exchange Current | $5.0$ | $\text{A}$ | Kinematic charge transfer rate at anode |
    | $\beta_c, \beta_a$ | Charge Transfer Coeff | $19.1$ | $\text{V}^{-1}$ | Kinetic symmetry factors ($\alpha F / R T$) |
    | $C_b$ | Bulk Cu Concentration | Sidebar | $\text{g/L}$ | Bulk concentration of $\text{Cu}^{2+}$ ions |
    | $T$ | Electrolyte Temperature | Sidebar | $^\circ\text{C}$ | Sets viscosity, density and Cu diffusivity |
    | $Q$ | Electrolyte Flow Rate | Sidebar | $\text{L/min}$ | Sets jet velocity, mass transfer and flow cooling |
    | $d_{hole}, p$ | Jet Hole Diameter, Pitch | $0.01, 0.02$ | $\text{m}$ | Cathode jet hole geometry |
    | $D_{Cu}$ | Cu Diffusivity | $6.0 \times 10^{-10}$ at 25 °C | $\text{m}^2/\text{s}$ | Stokes-Einstein temperature correction |
    | $F$ | Faraday Constant | $96485$ | $\text{C/mol}$ | Charge per mole of electrons |
    | $R_{gas}$ | Universal Gas Constant | $8.314$ | $\text{J/(mol}\cdot\text{K)}$ | Ideal gas constant |
    | $\rho_{sol}$ | Electrolyte Density | $1140$ | $\text{kg/m}^3$ | Density of 20% $\text{H}_2\text{SO}_4$ solution |
    | $C_p$ | Electrolyte Heat Capacity | $3500$ | $\text{J/(kg}\cdot\text{K)}$ | Specific heat capacity of the acid bath |
    """)

