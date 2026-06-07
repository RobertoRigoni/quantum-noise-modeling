import numpy as np
import scipy
import matplotlib.pyplot as plt

from math_setup import h, zero, one, pauli_x, pauli_y, pauli_z, hadamard
from math_setup import compute_prob_zero, compute_prob_one, compute_expectation_x, compute_expectation_y, compute_expectation_z, compute_expectation_hadamard
from math_setup import normalize, check_unitary, tensor_product, get_zero_probs, get_one_probs, get_prob_sums, psi_constant_hamiltonian_array, plot_constant_hamiltonian_zero_state, plot_constant_hamiltonian_one_state

# ---------------------
# Simulating Rabi Oscillations of |0> and |1> probabilities for a given Rabi frequency
# ---------------------


qubit_freq = 4.5e9 # Typically between 3 and 6 GHz

omega = 2 * np.pi * 50e6 # Typical Rabi Frequency

initial_psi = zero # |0>

constant_pauliz_hamiltonian = qubit_freq*0.5*pauli_z # Ensure that graphs are accurate when using pauli-z H

constant_paulix_hamiltonian = omega*0.5*pauli_x

times = np.linspace(0, 100e-9, 1000)

psi_of_t_vector = psi_constant_hamiltonian_array(initial_psi, times, constant_paulix_hamiltonian) # Generate array of state vectors that vary with time

psi_of_t_zero_probs = get_zero_probs(psi_of_t_vector)

psi_of_t_one_probs = get_one_probs(psi_of_t_vector)

#print(psi_of_t_vector)
#print(zero[0])

print(np.allclose(get_prob_sums(psi_of_t_vector), 1.0)) # Checks normalization of probabilities by ensuring sums of |0> and |1> probabilities always add to 1.

#print(times)

plot_constant_hamiltonian_zero_state(times, psi_of_t_zero_probs)

plot_constant_hamiltonian_one_state(times, psi_of_t_one_probs)