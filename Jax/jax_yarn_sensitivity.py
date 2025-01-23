import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"
jnp.trapz = jnp.trapezoid

def yarn_a(Vf, E11f, E22f, E33f, Em, Gm, vm, G12f, v12f, v23f, ha, MAa, Ly):
    Vm = 1 - Vf
    v12m = vm
    v23m = vm
    
    # Calculations for all yarns  
    S12f = -v12f/E11f
    S12m = -v12m/Em
    S11f = 1/E11f
    S11m = 1/Em
    S22f = 1/E22f
    S22m = 1/Em
    S21f = S12f
    S21m = S12m
    S23f = -v23f/E33f
    S23m = -v23m/Em
    a11 = Em/E11f
    a22 = 0.5*(1 + Em/E22f)
    a33 = a22
    a44 = a22
    a55 = 0.5*(1 + Gm/G12f)
    a66 = a55
    a12 = ((S12f-S12m)*(a11-a22))/(S11f-S11m)
    a13 = a12
    
    E11 = Vf*E11f + Vm*Em
    v12 = Vf*v12f + Vm*vm
    v13 = v12
    E22 = ((Vf+Vm*a11)*(Vf+Vm*a22))/ ((Vf+Vm*a11)*(Vf*S22f+ a22*Vm*S22m) + Vf*Vm*(S21m-S21f)*a12)
    E33 = E22
    G12 = Gm*((G12f+Gm)+Vf*(G12f-Gm))/((G12f+Gm)-Vf*(G12f-Gm))
    G13 = G12
    G23 = 0.5*(Vf+Vm*a22)/((Vf*(S22f-S23f))+Vm*a22*(S22m-S23m))
    v23 = 0.5*E22/G23 -1
    v32 = v23*E33/E22
    v21 = v12*E22/E11
    v31 = v13*E33/E11
    
    s11 = 1/E11
    s12 = -v12/E11
    s13 = -v13/E11
    s21 = s12
    s22 = 1/E22
    s23 = -v23/E33
    s31 = s13
    s32 = s23
    s33 = 1/E33
    s44 = 1/G23
    s55 = 1/G13
    s66 = 1/G12
    
    S = jnp.array([[s11, s12, s13, 0, 0, 0],
                  [s21, s22, s23, 0, 0, 0],
                  [s31, s32, s33, 0, 0, 0],
                  [0, 0, 0, s44, 0, 0],
                  [0, 0, 0, 0, s55, 0],
                  [0, 0, 0, 0, 0, s66]])
    
    C = jnp.linalg.inv(S)
    
    A = (MAa - ha) / 2
    L = Ly / 2
    alphaa = jnp.pi / 2
    
    # Define functions using JAX-compatible syntax
    def zprime(x):
        return A * jnp.sin(jnp.pi * x / L)
    
    def tanb(x):
        return (jnp.pi * A / L) * jnp.cos(jnp.pi * x / L)
    
    def m(x):
        return 1 / jnp.sqrt(1 + tanb(x)**2)
    
    def n(x):
        return tanb(x) / jnp.sqrt(1 + tanb(x)**2)
    
    # Define the integrand function
    def integrand(x):
        T1 = jnp.array([[m(x)**2, 0, n(x)**2, 0, 2 * m(x) * n(x), 0],
                        [0, 1, 0, 0, 0, 0],
                        [n(x)**2, 0, m(x)**2, 0, -2 * m(x) * n(x), 0],
                        [0, 0, 0, m(x), 0, -n(x)],
                        [-m(x) * n(x), 0, m(x) * n(x), 0, m(x)**2 - n(x)**2, 0],
                        [0, 0, 0, n(x), 0, m(x)]])
        
        T2 = jnp.array([[m(x)**2, 0, n(x)**2, 0, m(x) * n(x), 0],
                        [0, 1, 0, 0, 0, 0],
                        [n(x)**2, 0, m(x)**2, 0, -m(x) * n(x), 0],
                        [0, 0, 0, m(x), 0, -n(x)],
                        [-2 * m(x) * n(x), 0, 2 * m(x) * n(x), 0, m(x)**2 - n(x)**2, 0],
                        [0, 0, 0, n(x), 0, m(x)]])
        
        return jnp.dot(jnp.linalg.inv(T1), jnp.dot(C, T2))    
    
    # Generate x values for integration
    x_values = jnp.linspace(0, 2 * L, 10)
    
    # Compute the integrand values at each x
    integrand_values = jnp.array([integrand(x) for x in x_values])
    
    # Perform numerical integration using trapezoidal rule
    sigma_integral = 1/(2*L) * jnp.trapz(integrand_values, x_values, axis=0)
    
    # Define T3 and T4 matrices
    i = jnp.cos(alphaa)
    j = jnp.sin(alphaa)
    
    T3 = jnp.array([[i**2, j**2, 0, 0, 0, 2 * i * j],
                    [j**2, i**2, 0, 0, 0, -2 * i * j],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, i, -j, 0],
                    [0, 0, 0, j, i, 0],
                    [-i * j, i * j, 0, 0, 0, i**2 - j**2]])
    
    T4 = jnp.array([[i**2, j**2, 0, 0, 0, i * j],
                    [j**2, i**2, 0, 0, 0, -i * j],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, i, -j, 0],
                    [0, 0, 0, j, i, 0],
                    [-2 * i * j, 2 * i * j, 0, 0, 0, i**2 - j**2]])
    
    # Compute the final result
    sigmaprime = jnp.dot(jnp.linalg.inv(T3), jnp.dot(sigma_integral, T4))
            
    return sigmaprime

def yarn_b(Vf, E11f, E22f, E33f, Em, Gm, vm, G12f, v12f, v23f, ha, hb, Lx, rotation_angle):
    Vm = 1 - Vf
    v12m = vm
    v23m = vm
    
    # Calculations for all yarns  
    S12f = -v12f/E11f
    S12m = -v12m/Em
    S11f = 1/E11f
    S11m = 1/Em
    S22f = 1/E22f
    S22m = 1/Em
    S21f = S12f
    S21m = S12m
    S23f = -v23f/E33f
    S23m = -v23m/Em
    a11 = Em/E11f
    a22 = 0.5*(1 + Em/E22f)
    a33 = a22
    a44 = a22
    a55 = 0.5*(1 + Gm/G12f)
    a66 = a55
    a12 = ((S12f-S12m)*(a11-a22))/(S11f-S11m)
    a13 = a12
    
    E11 = Vf*E11f + Vm*Em
    v12 = Vf*v12f + Vm*vm
    v13 = v12
    E22 = ((Vf+Vm*a11)*(Vf+Vm*a22))/ ((Vf+Vm*a11)*(Vf*S22f+ a22*Vm*S22m) + Vf*Vm*(S21m-S21f)*a12)
    E33 = E22
    G12 = Gm*((G12f+Gm)+Vf*(G12f-Gm))/((G12f+Gm)-Vf*(G12f-Gm))
    G13 = G12
    G23 = 0.5*(Vf+Vm*a22)/((Vf*(S22f-S23f))+Vm*a22*(S22m-S23m))
    v23 = 0.5*E22/G23 -1
    v32 = v23*E33/E22
    v21 = v12*E22/E11
    v31 = v13*E33/E11
    
    s11 = 1/E11
    s12 = -v12/E11
    s13 = -v13/E11
    s21 = s12
    s22 = 1/E22
    s23 = -v23/E33
    s31 = s13
    s32 = s23
    s33 = 1/E33
    s44 = 1/G23
    s55 = 1/G13
    s66 = 1/G12
    
    S = jnp.array([[s11, s12, s13, 0, 0, 0],
                  [s21, s22, s23, 0, 0, 0],
                  [s31, s32, s33, 0, 0, 0],
                  [0, 0, 0, s44, 0, 0],
                  [0, 0, 0, 0, s55, 0],
                  [0, 0, 0, 0, 0, s66]])
    
    C = jnp.linalg.inv(S)
    
    A = (ha + hb) / 2
    L = jnp.abs((Lx / jnp.cos(jnp.pi / 2 - rotation_angle)) / 2 )
    alphab = jnp.pi / 2 - rotation_angle 
    
    # Define functions using JAX-compatible syntax
    def zprime(x):
        return A * jnp.sin(jnp.pi * x / L)
    
    def tanb(x):
        return (jnp.pi * A / L) * jnp.cos(jnp.pi * x / L)
    
    def m(x):
        return 1 / jnp.sqrt(1 + tanb(x)**2)
    
    def n(x):
        return tanb(x) / jnp.sqrt(1 + tanb(x)**2)
    
    # Define the integrand function
    def integrand(x):
        T1 = jnp.array([[m(x)**2, 0, n(x)**2, 0, 2 * m(x) * n(x), 0],
                        [0, 1, 0, 0, 0, 0],
                        [n(x)**2, 0, m(x)**2, 0, -2 * m(x) * n(x), 0],
                        [0, 0, 0, m(x), 0, -n(x)],
                        [-m(x) * n(x), 0, m(x) * n(x), 0, m(x)**2 - n(x)**2, 0],
                        [0, 0, 0, n(x), 0, m(x)]])
        
        T2 = jnp.array([[m(x)**2, 0, n(x)**2, 0, m(x) * n(x), 0],
                        [0, 1, 0, 0, 0, 0],
                        [n(x)**2, 0, m(x)**2, 0, -m(x) * n(x), 0],
                        [0, 0, 0, m(x), 0, -n(x)],
                        [-2 * m(x) * n(x), 0, 2 * m(x) * n(x), 0, m(x)**2 - n(x)**2, 0],
                        [0, 0, 0, n(x), 0, m(x)]])
        
        return jnp.dot(jnp.linalg.inv(T1), jnp.dot(C, T2))    
    
    # Generate x values for integration
    x_values = jnp.linspace(0, 2 * L, 10)
    
    # Compute the integrand values at each x
    integrand_values = jnp.array([integrand(x) for x in x_values])
    
    # Perform numerical integration using trapezoidal rule
    sigma_integral = 1/(2*L) * jnp.trapz(integrand_values, x_values, axis=0)
    
    # Define T3 and T4 matrices
    i = jnp.cos(alphab)
    j = jnp.sin(alphab)
    
    T3 = jnp.array([[i**2, j**2, 0, 0, 0, 2 * i * j],
                    [j**2, i**2, 0, 0, 0, -2 * i * j],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, i, -j, 0],
                    [0, 0, 0, j, i, 0],
                    [-i * j, i * j, 0, 0, 0, i**2 - j**2]])
    
    T4 = jnp.array([[i**2, j**2, 0, 0, 0, i * j],
                    [j**2, i**2, 0, 0, 0, -i * j],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, i, -j, 0],
                    [0, 0, 0, j, i, 0],
                    [-2 * i * j, 2 * i * j, 0, 0, 0, i**2 - j**2]])
    
    # Compute the final result
    sigmaprime = jnp.dot(jnp.linalg.inv(T3), jnp.dot(sigma_integral, T4))
            
    return sigmaprime

def stiff_mat(sigmaprime_a, sigmaprime_b1, sigmaprime_b2):
    
    Sm = jnp.array([[1/Em, -vm/Em, -vm/Em, 0, 0, 0],
                    [-vm/Em, 1/Em, -vm/Em, 0, 0, 0],
                    [-vm/Em, -vm/Em, 1/Em, 0, 0, 0],
                    [0, 0, 0, 1/Gm, 0, 0],
                    [0, 0, 0, 0, 1/Gm, 0],
                    [0, 0, 0, 0, 0, 1/Gm]])
                   
    Sm_inv = jnp.linalg.inv(Sm)
    Cvol = 0.2633 * sigmaprime_a + 0.2633 * sigmaprime_b1 + 0.2633 * sigmaprime_b2 + 0.2101 * Sm_inv
    Svol = jnp.linalg.inv(Cvol)    
    
    HomE11 = 1 / Svol[0, 0]
    HomE22 = 1 / Svol[1, 1]
    HomE33 = 1 / Svol[2, 2]
    HomG12 = 1 / Svol[5, 5]
    HomG23 = 1 / Svol[3, 3]
    HomG13 = 1 / Svol[4, 4]
    Homv12 = -Svol[0, 1] * HomE11
    Homv13 = -Svol[1, 2] * HomE22
    Homv23 = -HomE22 * Svol[2, 1]
    
    StiffMat = [HomE11, HomE22, HomE33, Homv12, Homv13, Homv23, HomG12, HomG13, HomG23]
          
    return HomE11, HomE22, HomE33, Homv12, Homv13, Homv23, HomG12, HomG13, HomG23

def forward():    
    sigmaprime_b1 = yarn_b(Vfb, E11f, E22f, E33f, Em, Gm, vm, G12f, v12f, v23f, ha, hb, Lx, rotation_b1)
    sigmaprime_b2 = yarn_b(Vfb, E11f, E22f, E33f, Em, Gm, vm, G12f, v12f, v23f, ha, hb, Lx, rotation_b2)
    sigmaprime_a = yarn_a(Vfa, E11f, E22f, E33f, Em, Gm, vm, G12f, v12f, v23f, ha, MAa, Ly)
      
    HomE11, HomE22, HomE33, Homv12, Homv13, Homv23, HomG12, HomG13, HomG23 = stiff_mat(sigmaprime_a, sigmaprime_b1, sigmaprime_b2)
    result = jnp.array([HomE11, HomE22, HomE33, Homv12, Homv13, Homv23, HomG12, HomG13, HomG23])
    
    return result

def sensitivity(inputs):
    
    # unpack inputs
    Vfa, Vfb, E11f, E22f, E33f, G12f, v12f, v23f, Em, Gm, vm, ha, hb, MAa, Lx, Ly, rotation_b1, rotation_b2 = inputs
   
    sigmaprime_b1 = yarn_b(Vfb, E11f, E22f, E33f, Em, Gm, vm, G12f, v12f, v23f, ha, hb, Lx, rotation_b1)
    sigmaprime_b2 = yarn_b(Vfb, E11f, E22f, E33f, Em, Gm, vm, G12f, v12f, v23f, ha, hb, Lx, rotation_b2)
    sigmaprime_a = yarn_a(Vfa, E11f, E22f, E33f, Em, Gm, vm, G12f, v12f, v23f, ha, MAa, Ly)
    
    HomE11, HomE22, HomE33, Homv12, Homv13, Homv23, HomG12, HomG13, HomG23 = stiff_mat(sigmaprime_a, sigmaprime_b1, sigmaprime_b2)
    
    return HomE11, HomE22, HomE33, Homv12, Homv13, Homv23, HomG12, HomG13, HomG23

if __name__ == "__main__":
    
    # For sensitivity calculation
    # TC275-1
    Vfa = jnp.float32(0.72)
    Vfb = jnp.float32(0.63)
    E11f = jnp.float32(230000.)
    E22f = jnp.float32(15000.)
    E33f = jnp.float32(15000.)
    Em = jnp.float32(4047.)
    Gm = jnp.float32(1516.)
    vm = jnp.float32(0.363)
    G12f = jnp.float32(24000.)
    v12f = jnp.float32(0.2)
    v23f = jnp.float32(0.491)

    ha = jnp.float32(0.299)
    hb = jnp.float32(0.234) 
    MAa = jnp.float32(0.510)
    Lx = jnp.float32(18.006)  # RVE unit cell dimension in X
    Ly = jnp.float32(10.030)  # RVE unit cell dimension in Y
    rotation_b1 = jnp.float32(jnp.pi/3)
    rotation_b2 = jnp.float32(-jnp.pi/3)

    # plotting sensitivities
    inputs = (Vfa, Vfb, E11f, E22f, E33f, G12f, v12f, v23f, Em, Gm, vm, ha, hb, MAa, Lx, Ly, rotation_b1, rotation_b2)
    sensitivities = jax.jacrev(sensitivity)(inputs)

    # Convert each array inside the tuple to a NumPy array and flatten them
    sensitivities_plot = [np.array(arr).flatten() for arr in sensitivities]

    # Stack the arrays vertically to create a 2D array
    sensitivities_2d = np.vstack(sensitivities_plot)

    # Plot names
    input_names = ['Vfa', 'Vfb', 'E11f', 'E22f', 'E33f', 'G12f', 'v12f', 'v23f', 'Em', 'Gm', 'vm', 'ha', 'hb', 'MAa', 'Lx', 'Ly', 'angle_b1', 'angle_b2']
    output_names = ['HomE11', 'HomE22', 'HomE33', 'Homv12', 'Homv13', 'Homv23', 'HomG12', 'HomG13', 'HomG23']

    # Improved function to plot sensitivities with larger fonts, higher quality, save images, and style the bars
    def plot_sensitivities(sensitivities, input_names, output_names, unit_names, save_dir='plots_2', y_axis_unit='',):
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(sensitivities.shape[1]):
            plt.figure(figsize=(6, 4), dpi=300)  
            x = np.arange(len(sensitivities[:, i])) 
            
            plt.bar(x, sensitivities[:, i], align='center', alpha=0.7, 
                    color='skyblue', edgecolor='black') 
            
            plt.xticks(x, output_names, rotation=45, fontsize=14) 
            
            # Include the unit in the y-axis label if provided
            ylabel = f'∂/∂{input_names[i]} ({unit_names[i]})' if unit_names[i] else f'∂/∂{input_names[i]}'
            plt.ylabel(ylabel, fontsize=14)  
            
            plt.grid(True)  
            plt.tight_layout()

            # Save the plot as a PNG file
            filename = f"{input_names[i]}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, format='png', dpi=300) 
            
            plt.show()

    sensitivities_MPa = jnp.concatenate((sensitivities_2d[0:3, :], sensitivities_2d[-3:, :]), axis=0)
    output_names_MPa = output_names[0:3] + output_names[-3:]
    unit_names = ['MPa', 'MPa', '', '', '', '', 'MPa', 'MPa', '', '', 'MPa', 'MPa/mm', 'MPa/mm', 'MPa/mm', 'MPa/mm', 'MPa/mm', 'MPa/rad', 'MPa/rad']

    plot_sensitivities(sensitivities_MPa, input_names, output_names_MPa, unit_names)
    