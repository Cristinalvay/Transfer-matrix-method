import numpy as np # Numerical methods
import matplotlib.pyplot as plt # Graphs

# Physics cte.
hbar = 1.0545718e-34  # Reduced Planck's constant (J*s)
m = 9.10938356e-31   # Electron mass (kg)
eV_to_J = 1.60218e-19  # eV to Joules conversion

# In order to make it a dynamic code
def get_user_input():
    
    # Prompt for energy and the initial/final potential
    E = float(input("Energy of the incident particle in a scattering experiment (in eV): ")) * eV_to_J  # Energy of particle in Joules
    V0 = float(input("Value of the potential at +/- infinity (in eV): ")) * eV_to_J  # Initial and final potential at extremes in Joules
    
    # Number of potential segments
    num_segments = int(input("Number of different potential segments: "))  
    
    # Lists to store potentials and lengths for each segment
    potentials = []
    lengths = []
    
    # Loop to gather potential and length for each segment
    for i in range(1, num_segments + 1):
        V = float(input(f"Potential V_{i} (in eV): ")) * eV_to_J  # Potential in Joules
        L = float(input(f"Length L_{i} (in nm): ")) * 1e-9  # Length in meters
        potentials.append(V)
        lengths.append(L)
    
    return E, V0, potentials, lengths


# wave number k for each value of V
def wave_number(E, V):
    if E > V:
        return np.sqrt(2 * m * (E - V)) / hbar # Scattering wave
    else:
        return 1j*np.sqrt(2 * m * (V - E)) / hbar  # Evanescence wave

# General Transfer matrix for a cte. potential
def transfer_matrix_0(k, L):
    return np.array([[np.exp(1j * k * L), 0], [0, np.exp(-1j * k * L)]], dtype=complex)

# General Transfer matrix for a step potential
def transfer_matrix_s(k_plus, k_minus):
    return 0.5 * np.array([[1 + k_minus / k_plus, 1 - k_minus / k_plus], [1 - k_minus / k_plus, 1 + k_minus / k_plus]], dtype=complex)

# Total transfer matrix function
def transfer_matrix(E, V0, potentials, lengths):
    M_total = np.eye(2, dtype=complex)  # Start with identity matrix 2x2

    k_1 = wave_number(E, potentials[0])  # Initial wave number for the first potential
    k_0 = wave_number(E, V0)
    M_total = transfer_matrix_s(k_1, k_0)  # Initial step transfer matrix

    # Loop over all potential segments and calculate the matrix product
    for i in range(len(potentials) - 1):
        
        k_plus = wave_number(E, potentials[i + 1])  # Wave number for potential[i+1]
        k_minus = wave_number(E, potentials[i])  # Wave number for potential[i]
        
        M_s = transfer_matrix_s(k_plus, k_minus)  # Transfer matrix for step (discontinuity)
        M_0 = transfer_matrix_0(k_minus, lengths[i])  # Transfer matrix for continuous potential

        M_total = M_s @ M_0 @ M_total  # Multiply the matrices together to update M_total
    
    # After the loop, we need to multiply by the last cte. and step matrix
    k_final = wave_number(E, potentials[-1])  # Last wave number for the final potential
    M_final_0 = transfer_matrix_0(k_final, lengths[-1])  # Last step transfer matrix
    M_final_s = transfer_matrix_s(k_0, k_final)  # Last step transfer matrix
    M_total = M_final_s @ M_final_0 @ M_total  # Final multiplication
    
    return M_total

# Reflection and transmission coefficients
def reflection_transmission_coefficients(M):
    r = M[1, 0] / M[1, 1]           # Reflection coefficient
    t = np.linalg.det(M) / M[1, 1]   # Transmission coefficient
    R = abs(r)**2                    # Reflectance (|r|^2)
    T = abs(t)**2                    # Transmittance (|t|^2)
    return R, T

# Main program
if __name__ == "__main__":

    E, V0, potentials, lengths = get_user_input()
    
    # Positions in which the potential changes
    x_positions = np.zeros(len(lengths) + 1)
    for i in range(1, len(lengths) + 1):
        x_positions[i] = x_positions[i - 1] + lengths[i - 1]
    
    # Calculations of predefined functions
    M_total = transfer_matrix(E, V0, potentials, lengths)
    R, T = reflection_transmission_coefficients(M_total)
    
    # Print results
    print(f"Matriz de transferencia: {M_total}")
    print(f"Reflectancia (R): {R}")
    print(f"Transmitancia (T): {T}")


    # Create the figure
    plt.figure(figsize=(12, 6))

    # Plotting the potential segments
    for i in range(len(potentials)):
        plt.plot([x_positions[i], x_positions[i + 1]], [potentials[i], potentials[i]], color='b', linewidth=3, label= 'Potential' if i==0 else "")

    # Adding first and last segment for the last potential value
    x_position_initial = x_positions[0] - (x_positions[-1]-x_positions[0])
    x_position_final = x_positions[-1] + (x_positions[-1]-x_positions[0])
    plt.plot([x_position_initial, x_positions[0]], [V0, V0], color='k', linewidth=3, label= 'V0')
    plt.plot([x_positions[-1], x_position_final], [V0, V0], color='k', linewidth=3)

    # Line at y=E
    plt.axhline(E, color='r', linewidth=2, linestyle='--', label= 'Energy level E') 

    # Title and labels
    plt.title('Potential V(x)')
    plt.xlabel('Position x (m)')
    plt.ylabel('Potential V (J)')
    plt.xlim(0-np.sum(lengths)*0.5 , np.sum(lengths)*1.5)  # X-axis limit based on the sum of lengths
    plt.ylim(0, max(max(potentials),E) + min(potentials))  # Y-axis limits
    plt.grid()
    plt.legend()
    plt.show()

