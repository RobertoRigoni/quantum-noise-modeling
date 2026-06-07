import numpy as np
import scipy
import matplotlib.pyplot as plt

from math_setup import h, zero, one, pauli_x, pauli_y, pauli_z, hadamard
from math_setup import compute_prob_zero, compute_prob_one, compute_expectation_x, compute_expectation_y, compute_expectation_z, compute_expectation_hadamard
from math_setup import normalize, check_unitary, tensor_product, get_zero_probs, get_one_probs, get_prob_sums, plot_constant_hamiltonian_zero_state, plot_constant_hamiltonian_one_state


class State():
    def __init__(self, alpha, beta, is_zero, is_one):
        self.alpha = alpha
        self.beta = beta
        self.is_zero = is_zero
        self.is_one = is_one

        psi_vector = np.array([[alpha], [beta]], dtype=complex)
        self.state = psi_vector

        if not np.allclose(np.linalg.norm(self.state), 1.0, 0, 0.001) :
            print("Probability sum is not 1; normalizing input")
            self.state = normalize(self.state)

        if self.is_zero and not self.is_one :
            self.state = zero
        
        elif not self.is_zero and self.is_one :
            self.state = one
        
        elif self.is_zero and self.is_one :
            print("Error assigning state value: both zero and one assigned true.\nSetting state to zero by default.")
            self.state = zero

    
    def get_state(self):
        return self.state
    
    
hamiltonian = 0.5 * (2 * np.pi * 50e6) * pauli_x

def evaluate_schro_eq(t, state, hamiltonian=hamiltonian) : 
    return hamiltonian @ ( (-1j) * state)

    

def rk4(t_n, dt, state_n, evaluate) :
    k1 = dt * evaluate(t_n, state_n)
    k2 = dt * evaluate(t_n + dt/2, state_n + 0.5*k1)
    k3 = dt * evaluate(t_n + dt/2, state_n + 0.5*k2)
    k4 = dt * evaluate(t_n + dt, state_n + k3)

    state_n_plus1 =  state_n + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

    return state_n_plus1


state_a = State(alpha=1j/np.sqrt(2), beta=-1j/np.sqrt(2), is_zero = True, is_one = False)

#print(state_a.get_state())
#print(evaluate_schro_eq(1, state_a.get_state())) # Test one hamiltonian

times = np.linspace(0, 100e-9, 1000)
steps = 1000
dt = 100e-9/(steps)

psi_history = [state_a.get_state()]


for i in range(steps) :
    next_psi = rk4(times[i], dt, psi_history[-1], evaluate_schro_eq)
    psi_history.append(next_psi)

del psi_history[-1] # Delete extra calculated psi entry


state_a_zero_probs = get_zero_probs(psi_history)
state_a_one_probs = get_one_probs(psi_history)

#plot_constant_hamiltonian_zero_state(times, state_a_zero_probs)
#plot_constant_hamiltonian_one_state(times, state_a_one_probs)