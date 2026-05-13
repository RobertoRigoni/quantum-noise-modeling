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
print("Hadamard\n", hadamard)


# ---------------------
# Expectation values and state probabilities
def compute_prob_zero(state) :
    return np.abs(np.power(state[0], 2))

def compute_prob_one(state) :
    return np.abs(np.power(state[1], 2))

def normalize(state) :
    return state * (1 / np.linalg.norm(state))

def check_unitary(matrix) :
    dim = len(matrix)
    return np.allclose(matrix.conj().T @ matrix, np.eye(dim, dtype=complex))

def tensor_product(a, b) :
    return np.kron(a, b)

def compute_expectation_x(state) :
    return np.real(np.vdot(state, pauli_x @ state))

def compute_expectation_y(state) :
    return np.real(np.vdot(state, pauli_y @ state))

def compute_expectation_z(state) :
    return np.real(np.vdot(state, pauli_z @ state))

def compute_expectation_hadamard(state) :
    return np.real((np.vdot(state, hadamard @ state)))


#print(compute_expectation_x(zero))
#print(compute_expectation_y(zero))
#print(compute_expectation_z(zero))
#print(compute_expectation_hadamard(zero))

# ---------------------
# Evolving a state over time

qubit_freq = 4.5e9 # Typically between 3 and 6 GHz

omega = 2 * np.pi * 50e6 # Typical Rabi Frequency

initial_psi = zero # |0>

constant_pauliz_hamiltonian = omega*0.5*pauli_z

constant_paulix_hamiltonian = omega*0.5*pauli_x

times = np.linspace(0, 100e-9, 1000)

def psi_constant_hamiltonian_array(psi_0, times, hamiltonian) : 

    states = []

    for t in times :
        hamil_matrix_exp = scipy.linalg.expm(-1j*t*hamiltonian)
        states.append(hamil_matrix_exp @ psi_0)
    
    return np.array(states)


def get_zero_state(state_array):
    zero_probabilities = []

    for i in range(len(state_array)) :
        probability = np.abs(state_array[i][0])**2
        zero_probabilities.append(probability)
    
    return zero_probabilities

def get_one_state(state_array):
    one_probabilities = []

    for i in range(len(state_array)) :
        probability = np.abs(state_array[i][1])**2
        one_probabilities.append(probability)
    
    return one_probabilities

def get_prob_sums(state_array) :
    sums = []
    zero_probs = get_zero_state(state_array)
    one_probs = get_one_state(state_array)

    for i in range(len(state_array)) :
        sums.append(zero_probs[i] + one_probs[i])

    return sums


psi_of_t = psi_constant_hamiltonian_array(initial_psi, times, constant_paulix_hamiltonian) # Generate array of state vectors that vary with time


def plot_constant_hamiltonian_state(times, probabilities_array):
    plt.plot(times, probabilities_array)
    plt.title("|0> State Probability under Constant Hamiltonian")
    plt.xlabel("Time")
    plt.ylabel("|0> State Probability")
    plt.show()


print(np.allclose(get_prob_sums(psi_of_t), 1.0)) # Checks normalization of probabilities by ensuring sums of |0> and |1> probabilities always add to 1.
#plot_constant_hamiltonian_state(times, get_zero_state(psi_of_t))
plot_constant_hamiltonian_state(times, get_one_state(psi_of_t))
#plot_constant_hamiltonian_state(times, get_prob_sums(psi_of_t))




