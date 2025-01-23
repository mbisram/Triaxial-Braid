import os
import jax
import jax.numpy as jnp
from jax import value_and_grad
import csv
import optax
import time

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

def obj_func(params):
    Vfa, Vfb, ha, hb = params
   
    sigmaprime_b1 = yarn_b(Vfb, E11f, E22f, E33f, Em, Gm, vm, G12f, v12f, v23f, ha, hb, Lx, rotation_b1)
    sigmaprime_b2 = yarn_b(Vfb, E11f, E22f, E33f, Em, Gm, vm, G12f, v12f, v23f, ha, hb, Lx, rotation_b2)
    sigmaprime_a = yarn_a(Vfa, E11f, E22f, E33f, Em, Gm, vm, G12f, v12f, v23f, ha, MAa, Ly)
      
    HomE11, HomE22, HomE33, Homv12, Homv13, Homv23, HomG12, HomG13, HomG23 = stiff_mat(sigmaprime_a, sigmaprime_b1, sigmaprime_b2)
    
    # set result here for the, target properties that you want (e.g. HomE11, HomE22)
    # result = jnp.array([HomE11, HomE22, HomE33, Homv12, Homv13, Homv23, HomG12, HomG13, HomG23])
    # result = jnp.array([HomE11, HomE22, HomE33, HomG12, HomG13, HomG23])
    # result = jnp.array([HomE11, HomE22, HomG12])
    result = jnp.array([HomE11, HomE22])
    
    # mean squared error loss function
    loss = jnp.mean((result - target) ** 2)
    # loss = jnp.mean(((result - target_mean) / target_std - normalized_target) ** 2)
    
    # Split the result into three segments, normalizing but separately for moduli, poisson's, and shear
    # result_first_segment = result[:3]
    # result_middle_segment = result[3:6]
    # result_last_segment = result[6:]
    # normalized_result_first_segment = (result_first_segment - first_mean) / first_std
    # normalized_result_middle_segment = (result_middle_segment - middle_mean) / middle_std
    # normalized_result_last_segment = (result_last_segment - last_mean) / last_std
    # normalized_result = jnp.concatenate([normalized_result_first_segment, normalized_result_middle_segment, normalized_result_last_segment])
    # loss = jnp.mean((normalized_result - normalized_target) ** 2)
    
    return loss

def update(params, opt_state):
    # Compute the loss and gradients
    loss, grads = value_and_grad(obj_func)(params)

    # Update the parameters using the optimizer
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    params = jax.tree_util.tree_map(lambda x: jnp.clip(x, 0.1, 0.9), params) # adding a clip to stay within "physical" domain

    return loss, params, opt_state

if __name__ == "__main__":
    
    # Initializing constants
    # TC275-1 material properties
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
    
    # # all the variables you actually want to optimize for
    # scalar = 0.5
    # Vfa = jnp.float32(0.72 *scalar)
    # Vfb = jnp.float32(0.63 *scalar)
    # ha = jnp.float32(0.299 *scalar)
    # hb = jnp.float32(0.234 *scalar)

    # # flat = 0.5
    # Vfa = jnp.float32(0.5)
    # Vfb = jnp.float32(0.5)
    # ha = jnp.float32(0.5)
    # hb = jnp.float32(0.5)

    # # isotropic forward all
    # Vfa = jnp.float32(0.710268378)
    # Vfb = jnp.float32(0.64053309)
    # ha = jnp.float32(0.33018145)
    # hb = jnp.float32(0.508989513)

    # # isotropic forward just e11 e22
    # Vfa = jnp.float32(0.673049212)
    # Vfb = jnp.float32(0.671194911)
    # ha = jnp.float32(0.253370225)
    # hb = jnp.float32(0.192679331)

    # # calibration
    # Vfa = jnp.float32(0.646291375)
    # Vfb = jnp.float32(0.599262118)
    # ha = jnp.float32(0.196647108)
    # hb = jnp.float32(0.218096197)

    params = jnp.array([Vfa, Vfb, ha, hb])

    # Target homogenized properties (TC275-1)
    # target = jnp.array([4.5721191e+04, 5.0474203e+04, 1.1229098e+04, 2.7199343e-01, 3.3305216e-01, 3.3305225e-01, 1.7697281e+04, 3.8809763e+03, 4.1473545e+03]) #forward
    # target = jnp.array([42260, 50080, 10720, 0.242, 0.375, 0.335, 15810, 3830, 3520]) #FEA
    # target = jnp.array([43400, 45500, 16500]) #experiment
    target = jnp.array([48097.697, 48097.697]) #10 int points, in-plane isotropic
    print(target)

    # Normalize the target variables
    # target_mean = jnp.mean(target)
    # target_std = jnp.std(target)
    # normalized_target = (target - target_mean) / target_std

    # Normalizing but separately for moduli, poisson's, and shear
    # first_segment = target[:3]
    # middle_segment = target[3:6]
    # last_segment = target[6:]
    # first_mean, first_std = jnp.mean(first_segment), jnp.std(first_segment)
    # middle_mean, middle_std = jnp.mean(middle_segment), jnp.std(middle_segment)
    # last_mean, last_std = jnp.mean(last_segment), jnp.std(last_segment)
    # normalized_first_segment = (first_segment - first_mean) / first_std
    # normalized_middle_segment = (middle_segment - middle_mean) / middle_std
    # normalized_last_segment = (last_segment - last_mean) / last_std
    # normalized_target = jnp.concatenate([normalized_first_segment, normalized_middle_segment, normalized_last_segment])
    # print(normalized_target)
    
    result = forward()
    print(f'result: {result}')

    # Define the optimizer (Adam with a learning rate of 0.001)
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)
    
    num_epochs = 1000
    csv_name = 'csv output filename here'  
    start_time = time.time()

    # Create the CSV file and write the header
    with open(f'{csv_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Vfa', 'Vfb', 'ha', 'hb', 'Loss']) # change these headers based on your input params being optimized
        
    # Optimization loop
    for i in range(num_epochs):
        loss, params, opt_state = update(params, opt_state)
        
        # Convert params tuple to list of scalars
        param_list = [param.item() for param in params]
        
        # Append the current epoch's parameters and loss to the CSV file
        with open(f'{csv_name}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i] + param_list + [loss.item()])
        
        if i % 1 == 0:
            print(f"Optimization progress: step {i} of {num_epochs}, {i/num_epochs*100:.2f}%, Loss: {loss.item()}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken for {num_epochs} epochs: {elapsed_time:.2f} seconds")

    with open(f'{csv_name}.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(['Total Time Elapsed (seconds)', elapsed_time])
