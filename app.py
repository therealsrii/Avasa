import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

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

# Backward compatibility helper for NumPy 2.x trapezoid / trapz
trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

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
N_sets = st.sidebar.slider("Number of Electrode Sets", min_value=1, max_value=100, value=45, step=1, help="Number of identical anode-cathode pairs operating in the bath.")
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
I_target = st.sidebar.slider("Current (A)", min_value=10, max_value=500, value=150, step=10, help="Target current applied to the electrowinning cell (default is 150 A, corresponding to 150 A/m² on a 1m² active plate area).")
kappa_cond = st.sidebar.slider("Electrolyte Conductivity (S/m)", min_value=10, max_value=120, value=70, step=5, help="Electrical conductivity of 20% H2SO4 with copper. Default is ~70 S/m.")

st.sidebar.subheader("Pulsing Parameters")
f_pulse = st.sidebar.slider("Pulsing Frequency (Hz)", min_value=1.0, max_value=60.0, value=30.0, step=1.0, help="Operating pulsing frequency of the current.")
D_pulse = st.sidebar.slider("Duty Cycle (D)", min_value=0.05, max_value=1.0, value=0.50, step=0.05, help="Operating duty cycle of the pulses.")

st.sidebar.subheader("Heat Transfer")
U_coeff = st.sidebar.slider("Heat Loss Coefficient U (W/m²K)", min_value=0.0, max_value=50.0, value=10.0, step=1.0, help="Heat transfer coefficient to the surrounding air. 0 = Adiabatic cell.")
t_run_min = st.sidebar.slider("Operation Duration (min)", min_value=5, max_value=120, value=20, step=5, help="Duration of electrowinning operation.")
t_run = t_run_min * 60.0 # Convert to seconds

# Hidden/Static parameters
rho_sol = 1140.0 # Density (kg/m^3)
m_sol = rho_sol * vol_sol  # Mass of solution (kg)
C_p = 3500.0     # Heat capacity of 20% H2SO4 (J/kg/K)
E_eq = 0.89      # Equilibrium cell potential (V)
E_tn = 1.15      # Thermoneutral voltage (V)

# Double-layer properties
C_c = 0.2        # Cathode double-layer capacitance (F)
C_a = 0.2        # Anode double-layer capacitance (F)
I0_c = 10.0      # Cathode exchange current (A)
I0_a = 5.0       # Anode exchange current (A)
beta_c = 19.1    # Cathode charge transfer coeff (V^-1)
beta_a = 19.1    # Anode charge transfer coeff (V^-1)

# Mass transport
C_b = 600.0      # Bulk copper concentration (mol/m^3)
tau_diff = 15.0  # Diffusion time constant (s)
delta = 1e-4     # Diffusion boundary layer thickness (m)
F = 96485.0      # Faraday constant (C/mol)
R_gas = 8.314    # Gas constant (J/mol/K)
T_ref = 303.15   # Reference Temperature (K)

# Calculated Resistance
R_sol = d_gap / (kappa_cond * A_plate)

# ==========================================
# SOLVER IMPLEMENTATION
# ==========================================

def solve_overpotentials_fast(I_p):
    """
    Computes steady-state overpotentials. Since double-layer time constant is
    on the order of microseconds, it is in steady state during the entire 'on' period.
    """
    if I_p <= 0:
        return 0.0, 0.0
    eta_c_on = (1.0 / beta_c) * np.arcsinh(I_p / (2 * I0_c))
    eta_a_on = (1.0 / beta_a) * np.arcsinh(I_p / (2 * I0_a))
    return eta_c_on, eta_a_on

def solve_concentration_overpotential_on_fast(f, D, I_p):
    """
    Analytically solves the linear diffusion boundary layer ODE for concentration overpotential.
    """
    T = 1.0 / f
    T_on = D * T
    
    K = 1.0 / (2 * F * A_plate * delta)
    E_1 = np.exp(-T_on / tau_diff)
    E_2 = np.exp(-(T - T_on) / tau_diff)
    
    C_1 = C_b - K * tau_diff * I_p
    
    # Boundary concentration Cs(0) in periodic steady state
    Cs_0 = C_b - K * tau_diff * I_p * (E_2 * (1.0 - E_1)) / (1.0 - E_1 * E_2 + 1e-15)
    
    # Sample points to integrate over the 'on' period
    t_vals = np.linspace(0, T_on, 50)
    Cs_vals = C_1 + (Cs_0 - C_1) * np.exp(-t_vals / tau_diff)
    Cs_vals = np.maximum(1e-4 * C_b, Cs_vals) # Cap to avoid non-positive concentration
    
    # Concentration overpotential (V)
    eta_conc_vals = -(R_gas * T_ref / (2 * F)) * np.log(Cs_vals / C_b)
    
    return trapezoid(eta_conc_vals, t_vals) / T_on

def calculate_cell_metrics(f, D, mode_select, N_sets=1):
    """
    Calculates cell voltages, powers, and temperature rise for the whole bath (N_sets).
    """
    if mode_select == "Constant Average Current":
        I_avg = float(I_target)
        I_p = I_avg / D
    else:
        I_p = float(I_target)
        I_avg = I_p * D
        
    if f == 0.0:
        # DC case
        eta_c = (1.0 / beta_c) * np.arcsinh(I_avg / (2 * I0_c))
        eta_a = (1.0 / beta_a) * np.arcsinh(I_avg / (2 * I0_a))
        K = 1.0 / (2 * F * A_plate * delta)
        Cs = np.maximum(1e-4 * C_b, C_b - K * tau_diff * I_avg)
        eta_conc = -(R_gas * T_ref / (2 * F)) * np.log(Cs / C_b)
        
        V_cell_on = E_eq + eta_a + eta_c + eta_conc + I_avg * R_sol
        Q_joule = N_sets * (I_avg**2 * R_sol)
        Q_over = N_sets * (I_avg * (eta_a + eta_c + eta_conc))
        Q_chem = N_sets * (I_avg * (E_eq - E_tn))
        Q_gen = Q_joule + Q_over + Q_chem
    else:
        # Pulsed case
        eta_c_on, eta_a_on = solve_overpotentials_fast(I_p)
        eta_conc_on = solve_concentration_overpotential_on_fast(f, D, I_p)
        
        V_cell_on = E_eq + eta_a_on + eta_c_on + eta_conc_on + I_p * R_sol
        
        # Joule heating: average current squared * R during on-time = D * I_p^2 * R = I_avg * I_p * R
        Q_joule = N_sets * (D * I_p**2 * R_sol)
        Q_over = N_sets * (I_avg * (eta_a_on + eta_c_on + eta_conc_on))
        Q_chem = N_sets * (I_avg * (E_eq - E_tn))
        Q_gen = Q_joule + Q_over + Q_chem
        
    # Temperature rise calculations with convective heat losses
    # Total heat loss area scales with number of cells in the tank
    A_loss = 2.0 * A_plate + (4.0 * N_sets * np.sqrt(A_plate) * d_gap)
    UA = U_coeff * A_loss
    mC = m_sol * C_p
    
    if UA == 0:
        dT = (Q_gen * t_run) / mC
    else:
        dT = (Q_gen / UA) * (1.0 - np.exp(-(UA / mC) * t_run))
        
    return V_cell_on, Q_joule, Q_over, Q_chem, Q_gen, dT

# ==========================================
# MAIN PAGE LAYOUT
# ==========================================

st.title("Copper Electrowinning Thermal Simulator")
st.markdown("""
This interactive simulation models the thermal dynamics of a pulsed-current copper electrowinning cell system.
You can adjust the parameters in the sidebar to see how cell geometry, electrical properties, and heat loss coefficients affect the temperature rise.
""")

# Calculate current active operating point metrics using the sidebar pulsing sliders
V_on, P_joule, P_over, P_chem, P_gen, dT_final = calculate_cell_metrics(f_pulse, D_pulse, mode, N_sets)

# Callout box for the whole bath forecast (N_sets)
st.markdown(f"""
<div class="metric-card" style="background-color: #fef2f2; border: 1px solid #fee2e2; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
    <div style="font-size: 0.875rem; font-weight: 700; color: #ef4444; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Whole Bath Thermal Forecast ({N_sets} Electrode Sets)</div>
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <div style="font-size: 2.25rem; font-weight: 800; color: #b91c1c; letter-spacing: -0.03em;">+{dT_final:.2f} °C</div>
            <div style="font-size: 0.85rem; color: #4b5563; margin-top: 2px;">Temperature Rise after <b>{t_run_min} min</b> of operation (at {f_pulse:.1f} Hz, {D_pulse*100:.0f}% D)</div>
        </div>
        <div style="text-align: right; min-width: 200px;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #09090b;">{P_gen/1000.0:.2f} kW</div>
            <div style="font-size: 0.85rem; color: #4b5563; margin-top: 2px;">Total Heat Generation Rate</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Peak Cell Voltage</div>
        <div class="metric-value">{V_on:.2f} V</div>
        <div class="metric-subtext">Single cell voltage (on-pulse)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">System Joule Heat</div>
        <div class="metric-value">{P_joule/1000.0:.2f} kW</div>
        <div class="metric-subtext">Total Ohmic resistance heat</div>
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
    freqs = np.linspace(1.0, 60.0, 25)
    duties = np.linspace(0.05, 1.0, 25)
    F_grid, D_grid = np.meshgrid(freqs, duties)
    
    # Calculate Z (Temperature Rise)
    Z_grid = np.zeros_like(F_grid)
    for i in range(len(duties)):
        for j in range(len(freqs)):
            _, _, _, _, _, dT = calculate_cell_metrics(F_grid[i, j], D_grid[i, j], mode, N_sets)
            Z_grid[i, j] = dT
            
    # Calculate DC for plotting at f=0 boundary if desired
    # For neatness, we keep the surface bound within 1-60 Hz but show DC as reference in text.
    
    # Create Plotly 3D Surface
    colorscale = 'Magma' if mode == "Constant Average Current" else 'Viridis'
    
    fig = go.Figure(data=[go.Surface(
        x=F_grid,
        y=D_grid,
        z=Z_grid,
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text="Temp Rise (°C)", font=dict(color="#09090b", size=12)),
            tickfont=dict(color="#27272a", size=10)
        ),
        hovertemplate=(
            "Frequency: %{x:.1f} Hz<br>" +
            "Duty Cycle: %{y:.2f}<br>" +
            "Temp Rise: %{z:.3f} °C<br>" +
            "<extra></extra>"
        )
    )])
    
    # Add a marker point on the surface representing the current operating point selected via the sliders
    fig.add_trace(go.Scatter3d(
        x=[f_pulse],
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
            "Frequency: %{x:.1f} Hz<br>" +
            "Duty Cycle: %{y:.2f}<br>" +
            "Temp Rise: %{z:.3f} °C<br>" +
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Whole Bath Temperature Rise after {t_run_min} min ({N_sets} Sets, Current: {I_target} A, Volume: {vol_sol_L} L, Mode: {mode})",
            font=dict(color="#09090b", size=14)
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text="Pulsing Frequency (Hz)", font=dict(color="#09090b", size=12)),
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
    if mode == "Constant Average Current":
        st.markdown(f"""
        - **Joule Heating Blowup**: Because the average current is fixed at **{I_target} A**, decreasing the duty cycle forces the peak current to rise as $I_{{peak}} = I_{{avg}}/D$. Ohmic heating increases quadratically with peak current ($I_{{peak}}^2 R$), resulting in a net Joule heating scaling of $1/D$. At $D = 0.05$, peak current is **{I_target/0.05:.0f} A**, causing a very steep temperature rise.
        - **Frequency Flatness**: The surface is nearly flat along the frequency axis. This occurs because the electrical double-layer charges in microseconds ($\tau \approx 17\\ \\mu\\text{{s}}$) and the concentration diffusion layer relaxes slowly ($\\tau_{{diff}} \\approx 15\\ \\text{{s}}$). Both systems settle into steady-state cycles quickly, making frequency thermally neutral in the $1-60\\ \\text{{Hz}}$ range.
        """)
    else:
        st.markdown(f"""
        - **Linear Heat Decrease**: Since the peak current is fixed at **{I_target} A**, reducing the duty cycle decreases the average current linearly ($I_{{avg}} = I_{{peak}} \\cdot D$). Both Joule heating ($D \\cdot I_{{peak}}^2 R$) and overpotential power drop linearly with $D$, causing the temperature rise to drop to nearly 0°C at very low duty cycles.
        - **Production Trade-off**: While a lower duty cycle reduces solution heating, it also reduces the copper deposition rate proportionally due to the lower average current.
        """)

with tab2:
    st.subheader("Mathematical and Physical Framework")
    st.markdown(r"""
    This simulator models the coupling of electrochemical reaction kinetics, mass transport, and thermal dynamics in a pulsed-current electrowinning cell system.

    ### 1. Cell Electrical Model and Overpotentials
    The transient voltage required to drive the electrowinning cell is modeled as:
    
    $$V_{cell}(t) = E_{eq} + \eta_a(t) + \eta_c(t) + \eta_{conc}(t) + I(t) R_{sol}$$

    Where the individual components represent:
    
    *   **Thermodynamic Equilibrium Potential ($E_{eq}$)**: The thermodynamic potential required for the net copper electrowinning reaction:
        $$\text{Cu}^{2+} + \text{H}_2\text{O} \rightarrow \text{Cu} + \frac{1}{2}\text{O}_2 + 2\text{H}^+$$
        
    *   **Activation Overpotentials ($\eta_a, \eta_c$)**: The kinetic barriers for charge transfer at the anode and cathode surfaces. These are governed by Butler-Volmer kinetics coupled with double-layer capacitance charging:
        $$C_c \frac{d\eta_c}{dt} = I(t) - 2 I_{0,c}\sinh(\beta_c \eta_c)$$
        $$C_a \frac{d\eta_a}{dt} = I(t) - 2 I_{0,a}\sinh(\beta_a \eta_a)$$
        
    *   **Concentration Overpotential ($\eta_{conc}$)**: The voltage loss due to copper ion depletion at the cathode surface. It is governed by the Nernst equation:
        $$\eta_{conc}(t) = -\frac{R_{gas} T_{ref}}{2F} \ln\left( \frac{C_s(t)}{C_b} \right)$$
        Where the cathode surface concentration $C_s(t)$ is driven by the electrochemical reaction rate and replenished by Fickian boundary-layer diffusion:
        $$\frac{dC_s}{dt} = \frac{C_b - C_s(t)}{\tau_{diff}} - \frac{I(t)}{2 F A_{plate} \delta}$$
        
    *   **Ohmic Resistance ($R_{sol}$)**: The resistance of the bulk sulfuric acid solution separating the electrodes:
        $$R_{sol} = \frac{d_{gap}}{\kappa_{cond} A_{plate}}$$

    ---

    ### 2. Thermal Energy Balance Model
    The transient temperature rise of the electrolyte bath is governed by the net heat generation rate $Q_{gen}$ and convective heat losses to the ambient environment:

    $$m_{sol} C_p \frac{dT_{sol}}{dt} = Q_{gen, total} - U A_{loss} (T_{sol} - T_{amb})$$

    Key aspects of the thermal model include:
    
    *   **Endothermic Reaction Enthalpy Subtraction**: The chemical reaction absorbs energy. The net rate of heat generation is calculated by subtracting the reaction enthalpy, represented as a thermoneutral voltage ($E_{tn} = 1.15\text{ V}$):
        $$Q_{gen, total}(t) = N_{sets} \cdot I(t) \cdot (V_{cell}(t) - E_{tn})$$
        
    *   **Analytical Solution for Temperature Rise**: Integrating the thermal ODE yields the solution temperature rise over the operation duration $t$:
        $$\Delta T(t) = \frac{Q_{gen, avg}}{U A_{loss}} \left( 1 - \exp\left( -\frac{U A_{loss}}{m_{sol} C_p} t \right) \right)$$
        Where $Q_{gen, avg}$ is the average heat generation rate over a pulsing period.
        
    *   **Heat Loss Area ($A_{loss}$)**: Calculated based on the geometry of a rectangular tank containing $N_{sets}$ plates separated by gap $d_{gap}$:
        $$A_{loss} = 2 A_{plate} + 4 N_{sets} \sqrt{A_{plate}} d_{gap}$$

    ---

    ### 3. Model Parameters and Constants Reference Table
    The physical and chemical parameters used to evaluate the electrowinning cell model are listed below:

    | Symbol | Parameter Description | Nominal Value | Unit | Physical Significance |
    | :--- | :--- | :--- | :--- | :--- |
    | $E_{eq}$ | Equilibrium Potential | $0.89$ | $\text{V}$ | Thermodynamic potential for Cu deposition / $\text{O}_2$ evolution |
    | $E_{tn}$ | Thermoneutral Potential | $1.15$ | $\text{V}$ | Cell voltage at which net reaction heat generation is zero |
    | $C_c, C_a$ | Double-Layer Capacitance | $0.2$ | $\text{F}$ | Cathode & Anode double-layer capacitance |
    | $I_{0,c}$ | Cathode Exchange Current | $10.0$ | $\text{A}$ | Kinematic charge transfer rate at cathode |
    | $I_{0,a}$ | Anode Exchange Current | $5.0$ | $\text{A}$ | Kinematic charge transfer rate at anode |
    | $\beta_c, \beta_a$ | Charge Transfer Coeff | $19.1$ | $\text{V}^{-1}$ | Kinetic symmetry factors ($\alpha F / R T$) |
    | $C_b$ | Bulk Cu Concentration | $600.0$ | $\text{mol/m}^3$ | Bulk concentration of $\text{Cu}^{2+}$ ions |
    | $\tau_{diff}$ | Diffusion Time Constant | $15.0$ | $\text{s}$ | Concentration boundary layer relaxation time |
    | $\delta$ | Boundary Layer Thickness | $10^{-4}$ | $\text{m}$ | Nernst diffusion boundary layer thickness |
    | $F$ | Faraday Constant | $96485$ | $\text{C/mol}$ | Charge per mole of electrons |
    | $R_{gas}$ | Universal Gas Constant | $8.314$ | $\text{J/(mol}\cdot\text{K)}$ | Ideal gas constant |
    | $\rho_{sol}$ | Electrolyte Density | $1140$ | $\text{kg/m}^3$ | Density of 20% $\text{H}_2\text{SO}_4$ solution |
    | $C_p$ | Electrolyte Heat Capacity | $3500$ | $\text{J/(kg}\cdot\text{K)}$ | Specific heat capacity of the acid bath |
    """)

