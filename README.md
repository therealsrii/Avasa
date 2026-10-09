# Copper Electrowinning & Thermal Simulator

This repository contains physical, electrochemical, hydraulic, and thermal models for a pilot-scale copper Electrowinning (EW) and Solvent Extraction (SX) circuit. The project is focused on modeling and optimizing pulsed current electrodeposition to achieve high current efficiency while managing the thermal limits of the system.

## Project Structure

* **`pulsed_ew_model.py`**: The core physical module containing:
  * Dynamic viscosity, density, and Cu diffusivity calculator as functions of temperature.
  * Sherwood-number-based mass transfer coefficient ($k$) solver for cathode jet holes and stagnant dead zones.
  * Transient boundary layer diffusion solver under pulsed current ($D_{pulse}$, $f_{pulse}$) including depletion timing ($t_{deplete}$) and concentration overpotential calculations.
  * Hydraulic pressure drop estimators (Orifice Loss model and Modified Ergun equation).
  * Solvent Extraction (SX) flow rate calculator.
  * Multi-component heat generation model (Ohmic heating, anode/cathode activation, concentration overpotential, and chemical reaction terms).
* **`optimize_ew_safe.py`**: A differential evolution optimization script that maximizes copper production rate subject to two constraints:
  * Current efficiency $\ge 80\%$.
  * Electrolyte steady-state operating temperature under continuous flow $\le 50\text{ }^\circ\text{C}$ (safe materials limit).
* **`optimize_ew.py`**: Performance-only optimization script without safe thermal constraints.
* **`app.py`**: An interactive Streamlit web dashboard to simulate, visualize, and forecast cell voltages, heat generation, current efficiency, and temperature rises in the electrowinning bath. All physics comes from `pulsed_ew_model.py`, so the dashboard and the optimizers agree. It offers two thermal models: a closed bath heating up over time, and continuous flow at steady state. It also checks the operating point against the rectifier's current and voltage rating (default 4500 A / 6 V, with the electrode sets wired in parallel). For copper powder production it shows the deposit regime (applied vs. limiting current density in the jet and dead zones), hydrogen and oxygen evolution rates with an indicative ventilation requirement, and yearly copper yield.
* **`simulate_heating.py`**: A transient solver using SciPy to simulate electrowinning cell temperatures over time.

---

## Getting Started

### Prerequisites

You need Python 3.x along with the following libraries:
* `numpy`
* `scipy`
* `plotly`
* `streamlit`
* `matplotlib`

You can install dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Simulator Web App
To start the interactive Streamlit dashboard:
```bash
streamlit run app.py
```

### Running the Safe Optimizer
To find the optimal operating parameters under safety limits (max 50% duty cycle, max 50 °C steady-state temperature):
```bash
python optimize_ew_safe.py
```

---

## Physical Modeling Details

### Transient Concentration Depletion
Under pulsed current, the peak current density during the "on" pulse ($J_{peak} = J_{avg} / D_{pulse}$) can exceed the local mass transfer limit ($J_{lim} = z F k C_{bulk}$). The model solves the linear boundary layer diffusion ODE:
$$\frac{d C_s}{dt} = \frac{k(C_{bulk} - C_s)}{\delta} - \frac{J(t)}{z F \delta}$$
If the surface concentration depletes to zero ($C_s(t) = 0$) before the end of the pulse ($t_{deplete} < T_{on}$), the deposition current density is capped at the diffusion limit. This transient depletion is solved analytically in `pulsed_ew_model.py` to calculate the exact current efficiency of both jet and dead zones.

### Thermal Balance
The steady-state temperature of the cell container under continuous flow is modeled as:
$$T_{ss} = \frac{Q_{gen} + U A_{loss} T_{amb} + \dot{m} C_p T_{in}}{U A_{loss} + \dot{m} C_p}$$
where $\dot{m} C_p$ is the heat capacity rate of the electrolyte flow ($350\text{ L/min}$ provides $23.27\text{ kW/K}$ of cooling capacity), which dominates over convective loss ($UA \approx 60\text{ W/K}$).
