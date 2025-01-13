#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import display, clear_output
import ipywidgets as widgets


# In[24]:


class DonnanSimulation:
    def __init__(self):
        # System constants - adjusted for more visible effects
        self.R = 8.314 # Universal gas constant (J/mol·K) - used in osmotic pressure calculations
        self.T = 310.15 # Temperature in Kelvin (equivalent to 37°C, normal body temperature)
        self.F = 96485 # Faraday constant (C/mol) - used in membrane potential calculations
        self.P_w = 1e-10  # Water permeability coefficient (m/s) - how easily water crosses the capillary membrane
        self.A = 1e-4 # Surface area of the membrane (m²) - representative area of capillary wall in contact with tissue
        
        # Initial volumes
        self.V_i = 1.0e-6 # Initial interstitial fluid volume (m³) - fluid volume in tissue space
        self.V_c = 1.2e-6 # Initial capillary volume (m³) - blood vessel volume
        self.total_volume = self.V_i + self.V_c # Sum of interstitial and capillary volumes (conservation of total fluid)
        
        # Initial ion concentrations (mol/m³) - adjusted for better visibility
        self.Na_c = 145 # Capillary sodium - normal blood sodium level
        self.Na_i = 142 # Interstitial sodium - slightly lower than blood
        self.Cl_c = 110 # Capillary chloride - normal blood chloride level
        self.Cl_i = 108 # Interstitial chloride - slightly lower than blood
        
        # Fixed interstitial protein (mol/m³)
        self.P_i = 0.2  # Lowered to create bigger gradient potential

    def assess_edema_risk(self, volume_change, time_constant):
        if volume_change > 15:
            risk = "High"
            explanation = "Significant volume increase indicates high risk of edema formation"
            color = 'red'
        elif volume_change > 10:
            risk = "Moderate"
            explanation = "Volume increase suggests potential for edema development"
            color = 'orange'
        elif volume_change > 5:
            risk = "Low"
            explanation = "Minor volume changes indicate low risk of edema"
            color = 'yellow'
        else:
            risk = "Normal"
            explanation = "Volume changes within normal physiological range"
            color = 'green'
            
        if time_constant < 1000:
            speed = "rapid"
        else:
            speed = "gradual"
            
        return risk, explanation, color, speed

    def calculate_osmotic_pressure(self, c1, c2):
        return self.R * self.T * (c1 - c2)

    def system_dynamics(self, t, state, P_c):
        V_i, Na_i, Cl_i = state
        
        V_c = self.total_volume - V_i
        
        if V_i <= 0.1e-6 or V_i >= 1.9e-6:
            return [0, 0, 0]
            
        Na_c = self.Na_c * self.V_c / V_c
        Cl_c = self.Cl_c * self.V_c / V_c
        
        pi_protein = self.calculate_osmotic_pressure(P_c, self.P_i) * 0.5
        pi_ions = self.calculate_osmotic_pressure(Na_c + Cl_c, Na_i + Cl_i) * 0.3
        
        J_v = self.P_w * self.A * (pi_protein + pi_ions) * np.exp(-abs(V_i - self.V_i) / self.V_i)
        
        dV_i_dt = J_v
        dNa_i_dt = -Na_i * J_v / V_i + 0.05 * (Na_c - Na_i)
        dCl_i_dt = -Cl_i * J_v / V_i + 0.05 * (Cl_c - Cl_i)
        
        return [dV_i_dt, dNa_i_dt, dCl_i_dt]

    def simulate(self, P_c, t_span=1800):  # 30 minutes
        y0 = [self.V_i, self.Na_i, self.Cl_i]
        t = np.linspace(0, t_span, 300)
        
        solution = solve_ivp(
            lambda t, y: self.system_dynamics(t, y, P_c),
            [0, t_span],
            y0,
            t_eval=t,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )
        
        return solution.t, solution.y[0], solution.y[1], solution.y[2]



# In[25]:


def create_interactive_simulation():
    sim = DonnanSimulation()
    
    protein_slider = widgets.FloatSlider(
        value=0.8,
        min=0.2,
        max=10.0,
        step=0.2,
        description='Capillary Protein:',
        style={'description_width': 'initial'},
        layout={'width': '50%'}
    )
    
    output_plots = widgets.Output()
    output_text = widgets.Output()
    
    def update(change):
        P_c = change['new']
        t, V_i, Na_i, Cl_i = sim.simulate(P_c)
        
        volume_change_percent = ((V_i[-1] - V_i[0]) / V_i[0]) * 100
        target_volume = V_i[0] + 0.63 * (V_i[-1] - V_i[0])
        time_constant = t[np.argmin(np.abs(V_i - target_volume))]
        
        # Get risk assessment
        risk, explanation, color, speed = sim.assess_edema_risk(volume_change_percent, time_constant)
        
        with output_plots:
            clear_output(wait=True)
            
            fig = plt.figure(figsize=(12, 10))
            gs = plt.GridSpec(3, 2, figure=fig)
            
            # Main volume plot
            ax1 = fig.add_subplot(gs[0, :])
            volume_percent = ((V_i - V_i[0]) / V_i[0]) * 100
            ax1.plot(t/60, volume_percent, 'b-', linewidth=2)
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Volume Change (%)')
            ax1.set_title('Tissue Volume Change Over Time')
            ax1.grid(True)
            
            # Ion concentration changes
            ax2 = fig.add_subplot(gs[1, :])
            ax2.plot(t/60, Na_i - sim.Na_i, 'r-', label='ΔNa⁺', linewidth=2)
            ax2.plot(t/60, Cl_i - sim.Cl_i, 'g-', label='ΔCl⁻', linewidth=2)
            ax2.set_xlabel('Time (minutes)')
            ax2.set_ylabel('Ion Change (mol/m³)')
            ax2.set_title('Ion Concentration Changes')
            ax2.grid(True)
            ax2.legend()

            # Rate of volume change
            ax3 = fig.add_subplot(gs[2, 0])
            dV_dt = np.gradient(volume_percent, t/60)
            ax3.plot(t/60, dV_dt, 'purple', linewidth=2)
            ax3.set_xlabel('Time (minutes)')
            ax3.set_ylabel('Rate (%/min)')
            ax3.set_title('Rate of Volume Change')
            ax3.grid(True)

            # Final state diagram
            ax4 = fig.add_subplot(gs[2, 1])
            ax4.axis('equal')
            ax4.set_xlim(-1, 1)
            ax4.set_ylim(-1, 1)
            
            vessel = plt.Circle((0, 0), 0.3, color='red', alpha=0.3)
            tissue = plt.Circle((0, 0), 0.8, color='blue', alpha=0.2)
            ax4.add_patch(vessel)
            ax4.add_patch(tissue)
            
            if volume_change_percent > 0:
                ax4.arrow(0.3, 0, 0.2, 0, head_width=0.1, head_length=0.1, fc='b', ec='b', alpha=0.6)
            
            ax4.text(-0.2, 0, f'P_c={P_c:.1f}', fontsize=10)
            ax4.text(0.4, 0, f'ΔV={volume_change_percent:.1f}%', fontsize=10)
            
            ax4.set_title('Final State')
            ax4.axis('off')
            
            plt.tight_layout()
            display(fig)
            plt.close()
        
        with output_text:
            clear_output(wait=True)
            print(f"\nProtein Gradient Analysis:")
            print(f"------------------------")
            print(f"Capillary protein: {P_c:.1f} mol/m³")
            print(f"Interstitial protein: {sim.P_i:.1f} mol/m³")
            print(f"Gradient: {P_c - sim.P_i:.1f} mol/m³")
            print(f"\nVolume Changes:")
            print(f"------------------------")
            print(f"Total volume change: {volume_change_percent:.1f}%")
            print(f"Maximum rate: {np.max(np.abs(dV_dt)):.2f}%/min")
            print(f"Time Constant: {time_constant/60:.1f} minutes")
            print(f"\nIon Changes:")
            print(f"------------------------")
            print(f"Na⁺ change: {Na_i[-1] - sim.Na_i:.1f} mol/m³")
            print(f"Cl⁻ change: {Cl_i[-1] - sim.Cl_i:.1f} mol/m³")
            print(f"\nRisk Assessment:")
            print(f"------------------------")
            print(f"Edema Risk: {risk} ({color})")
            print(f"Assessment: {explanation}")
            print(f"Rate of Change: Changes occur at a {speed} rate")
    
    protein_slider.observe(update, names='value')
    update({'new': protein_slider.value})
    
    display(widgets.VBox([
        protein_slider,
        output_plots,
        output_text
    ]))

create_interactive_simulation()


# In[ ]:




