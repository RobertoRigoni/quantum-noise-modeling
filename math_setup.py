import numpy as np
import scipy
import matplotlib.pyplot as plt

# ---------------------
# Basis states, gates, and constants

h = (6.62e-34)/(2*np.pi)

zero = np.array([[1], [0]], dtype=complex)

one = np.array([[0], [1]], dtype=complex)

pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)

pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)

pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Define Hadamard gate
hadamard = (1/np.sqrt(2)) * np.array([[1, 1], 
                                      [1, -1]], dtype=complex)


# ---------------------
# Expectation values and state probabilities
# ---------------------

def compute_prob_zero(state) :
    return np.abs(np.power(state[0], 2))

def compute_prob_one(state) :
    return np.abs(np.power(state[1], 2))

def compute_expectation_x(state) :
    return np.real(np.vdot(state, pauli_x @ state))

def compute_expectation_y(state) :
    return np.real(np.vdot(state, pauli_y @ state))

def compute_expectation_z(state) :
    return np.real(np.vdot(state, pauli_z @ state))

def compute_expectation_hadamard(state) :
    return np.real((np.vdot(state, hadamard @ state)))

def normalize(state) :
    return state * (1 / np.linalg.norm(state))

def check_unitary(matrix) :
    dim = len(matrix)
    return np.allclose(matrix.conj().T @ matrix, np.eye(dim, dtype=complex))

def tensor_product(a, b) :
    return np.kron(a, b)

def get_zero_probs(state_array) :
    zero_probabilities = []

    for i in range(len(state_array)) :
        probability = np.abs(state_array[i][0])**2
        zero_probabilities.append(probability)
    
    return zero_probabilities

def get_one_probs(state_array) :
    one_probabilities = []

    for i in range(len(state_array)) :
        probability = np.abs(state_array[i][1])**2
        one_probabilities.append(probability)
    
    return one_probabilities

def get_prob_sums(state_array) :
    sums = []
    zero_probs = get_zero_probs(state_array)
    one_probs = get_one_probs(state_array)

    for i in range(len(state_array)) :
        sums.append(zero_probs[i] + one_probs[i])

    return sums

def psi_constant_hamiltonian_array(psi_0, times, hamiltonian) : 

    states = []

    for t in times :
        hamil_matrix_exp = scipy.linalg.expm(-1j*t*hamiltonian)
        states.append(hamil_matrix_exp @ psi_0)
    
    return np.array(states)


def plot_constant_hamiltonian_zero_state(times, probabilities_array) :
    plt.plot(times, probabilities_array)
    plt.title("|0> State Probability under Constant Hamiltonian")
    plt.xlabel("Time")
    plt.ylabel("State Probability")
    plt.show()

def plot_constant_hamiltonian_one_state(times, probabilities_array) :
    plt.plot(times, probabilities_array)
    plt.title("|1> State Probability under Constant Hamiltonian")
    plt.xlabel("Time")
    plt.ylabel("State Probability")
    plt.show()