from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit import QuantumCircuit
import numpy as np
from qiskit.quantum_info import average_gate_fidelity
from scipy.optimize import curve_fit

import warnings
from typing import List, Union

from .utils import cyclic_permutations


def generate_random_pauli_probs(num_qubits: int, seed: int = None, p: float = 5e-3) -> list[tuple[float, float, float]]:
    """
    Generates random Pauli error probabilities (px, py, pz) for each qubit.
    Total error probability is sampled from Uniform[0, p].
    The error is distributed among X, Y, Z via flat Dirichlet distribution.
    """
    rng = np.random.default_rng(seed)
    probs = []
    for _ in range(num_qubits):
        p_i = rng.uniform(0, p)
        
        # 3. Uniformly split p_err_total among X, Y, Z using Dirichlet
        weights = rng.dirichlet(alpha=[1, 1, 1])
        px, py, pz = p_i * weights
        
        probs.append((px, py, pz))
    return probs


def create_cz_pauli_noise_model(pauli_probs: list[tuple[float, float, float]]) -> NoiseModel:
    """
    Creates a Qiskit NoiseModel where independent local Pauli noise is applied 
    after every CZ gate on each participating qubit.
    
    Args:
        pauli_probs: List of (px, py, pz) tuples, one per qubit.
        
    Returns:
        qiskit_aer.noise.NoiseModel configured with the specified CZ-local Pauli noise.
    """
    num_qubits = len(pauli_probs)
    noise_model = NoiseModel()

    # Precompute 1-qubit Pauli error channels for each qubit
    single_qubit_errors = []
    for px, py, pz in pauli_probs:
        p_i = 1.0 - (px + py + pz)  # Automatically matches 1 - p_i from generator
        err = pauli_error([('X', px), ('Y', py), ('Z', pz), ('I', p_i)])
        single_qubit_errors.append(err)

    # Attach tensor-product errors to all possible CZ gate pairs
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            combined_error_ij = single_qubit_errors[i] ^ single_qubit_errors[j]
            combined_error_ji = single_qubit_errors[j] ^ single_qubit_errors[i]
            noise_model.add_quantum_error(combined_error_ij, 'cz', [i, j])
            noise_model.add_quantum_error(combined_error_ji, 'cz', [j, i])

    return noise_model

def sum_gate_errors(transpiled_circuit, noise_model):
    """
    Computes the sum of probabilities of error over all CZ gates in the transpiled circuit.

    Args:
        circuit (QuantumCircuit): The input quantum circuit.
        backend (IBMQBackend): The target backend for transpilation.

    Returns:
        float: The sum of probabilities of error for all CZ gates in the transpiled circuit.
    """

    total_error = 0.0
    
    for instruction in transpiled_circuit.data:
        gate = instruction.operation
        gate_name = gate.name
        qubits = [transpiled_circuit.find_bit(qubit)[0] for qubit in instruction.qubits]  # Qubit indices the gate acts on

        # Get the CZ gate error from the noise model
        if gate_name == 'cz':
            try:
                error_obj = noise_model._local_quantum_errors[gate_name][tuple(qubits)]
            except:
                error_obj = noise_model._local_quantum_errors[gate_name][(tuple(qubits)[1], tuple(qubits)[0])]
            
            error = 1 - average_gate_fidelity(error_obj)
            total_error += error
    return total_error


def compute_exact_noisy_expectation(circuit, observables, noise_model):
    """
    Computes exact noisy expectation values for a list of observables
    using density matrix simulation with the provided noise model.
    
    Args:
        circuit: qiskit.QuantumCircuit
        observables: list of qiskit.quantum_info.SparsePauliOp (or Hermitian matrices)
        noise_model: qiskit_aer.noise.NoiseModel
        
    Returns:
        list[float]: Exact expectation values for each observable
    """
    # 1. Clone circuit and append density matrix save instruction
    circ_dm = circuit.copy()
    circ_dm.save_density_matrix(label="final_rho")
    
    # 2. Configure exact density matrix simulator
    sim = AerSimulator(method='density_matrix', noise_model=noise_model)
    
    # 3. Run simulation (method='density_matrix' computes deterministically, ignoring shots)
    result = sim.run(circ_dm).result()
    
    # 4. Extract the final density matrix
    rho = result.data(0)["final_rho"]  # Returns a qiskit.quantum_info.DensityMatrix
    
    # 5. Compute Tr(O @ ρ) for each observable
    return [rho.expectation_value(obs).real for obs in observables]

def clp_zne_under_pauli_noise(circuit, observables):
    """
    Zero-Noise Extrapolation Mitigation using Cyclic Layout Permutations (https://arxiv.org/pdf/2511.02901) for local Pauli noise.
    """
    num_qubits = circuit.num_qubits
    num_cycles = 4

    pauli_probs_per_cycle = [generate_random_pauli_probs(num_qubits, seed=i) for i in range(num_cycles)]
    
    cyclically_permuted_pauli_probs = []
    for pauli_probs in pauli_probs_per_cycle:
        cyclically_permuted_pauli_probs.extend(cyclic_permutations(pauli_probs))
    
    noise_models = [create_cz_pauli_noise_model(pauli_probs) for pauli_probs in cyclically_permuted_pauli_probs]

    evals_noisy = np.array([compute_exact_noisy_expectation(circuit, observables,  noise_model) for noise_model in noise_models]).T

    
    X = np.array(pauli_probs_per_cycle).mean(axis=1)
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

    evals_mitigated = []
    for obs_idx in range(len(observables)):
        y_data = evals_noisy[obs_idx]
        y = y_data.reshape((num_cycles, -1)).mean(axis=1)
        # Perform linear fit
        coeffs, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        # The first coefficient is the intercept (noise -> 0 limit)
        evals_mitigated.append(coeffs[0])
        
    return evals_mitigated, evals_noisy, X

def vanilla_lp_zne_under_pauli_noise(circuit, observables, num_perms, num_cycles=4, seed=42, is_fully_connected=False):
    """
    Zero-Noise Extrapolation Mitigation using random circuit layout permutations (https://arxiv.org/pdf/2307.11156) for local Pauli noise.
    """
    rng = np.random.default_rng(seed)
    num_qubits = circuit.num_qubits

    pauli_probs_per_cycle = [generate_random_pauli_probs(num_qubits, seed=i) for i in range(num_cycles)]

    if is_fully_connected:
        chosen_permutations = [rng.permutation(num_qubits) for _ in range(num_perms)]
        chosen_cycles = rng.choice(num_cycles, size=num_perms, replace=True).tolist()
        pauli_probs_of_chosen_qubits = [[pauli_probs_per_cycle[cycle][i] for i in perm] 
                                        for cycle, perm in zip(chosen_cycles, chosen_permutations)]
    else:
        indices = list(range(num_qubits))
        cyclic_permutations_list = []
        cyclic_permutations_list.extend(cyclic_permutations(indices))
        cyclic_permutations_list.extend(cyclic_permutations(indices[::-1]))

        chosen_permutations_indices = rng.choice(len(cyclic_permutations_list), size=num_perms, replace=True).tolist()
        chosen_permutations = [cyclic_permutations_list[i] for i in chosen_permutations_indices]
        chosen_cycles = rng.choice(num_cycles, size=num_perms, replace=True).tolist()
        pauli_probs_of_chosen_qubits = [[pauli_probs_per_cycle[cycle][i] for i in perm] 
                                        for cycle, perm in zip(chosen_cycles, chosen_permutations)]

    noise_models = [create_cz_pauli_noise_model(pauli_probs) for pauli_probs in pauli_probs_of_chosen_qubits]

    evals_noisy = np.array([compute_exact_noisy_expectation(circuit, observables,  noise_model) for noise_model in noise_models]).T

    X = np.array([sum_gate_errors(circuit, noise_model) for noise_model in noise_models]).T
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

    evals_mitigated = []
    for obs_idx in range(len(observables)):
        y = evals_noisy[obs_idx]

        coeffs, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        # The first coefficient is the intercept (noise -> 0 limit)
        evals_mitigated.append(coeffs[0])
        
    return evals_mitigated, evals_noisy, X


def unitary_folding_zne_under_pauli_noise(circuit, observables, folding_method='gate', extrapolation_method='exponential'):
    """
    Implements Digital Unitary Folding Zero-Noise Extrapolation with scale factors [1, 3, 5, 7].
    
    :param circuit: quantum circuit.
    :param observables: list of observables.
    :param layout: qubit layout.
    :param backend: backend to run the circuit on.
    :param noise_model: noise model used in simulation.
    :param folding_method: method to use for noise amplification. Possible values are 'gate' and 'circuit' 
           to perform unitary gate folding and unitary circuit folding respectivly. Default is 'gate'.
    :param extrapolation_method: function to use for extrapolation. Possible values are 'exponential' and 'linear'
            to fit f(x) = A + B * C^x and f(x) = A + B * x respectively. Default is 'exponential'.
    """
    num_qubits = circuit.num_qubitss

    scaled_circuits = []
    scale_factors = [1, 3, 5, 7]
    for scale_factor in scale_factors:
        scaled_circuits.append(fold_circuit(circuit, scale_factor, folding_method=folding_method))


    pauli_probs = generate_random_pauli_probs(num_qubits, seed=3)
    noise_model = create_cz_pauli_noise_model(pauli_probs)

    evals_noisy = [compute_exact_noisy_expectation(scaled_circ, observables, noise_model) for scaled_circ in scaled_circuits]
    evals_noisy = np.array(evals_noisy).T

    if extrapolation_method == 'exponential':
        evals_mitigated = zne_exponential_mitigation(evals_noisy)
    elif extrapolation_method == 'linear':
        evals_mitigated = zne_linear_mitigation(evals_noisy)
    else:
        raise ValueError(r"extrapolation_method should be 'linear' or 'exponential'.")

    return evals_mitigated, evals_noisy

def fold_circuit(circuit: QuantumCircuit, scale_factor: float, folding_method='gate') -> QuantumCircuit:
    """
    Fold a quantum circuit to amplify noise. Removes all end circuit measurments.
    
    Args:
        circuit: Original quantum circuit
        scale_factor: Noise amplification factor (must be odd: 1, 3, 5, ...)
        folding_method: method to use for noise amplification. Possible values are 'gate' and 'circuit' 
        to perform unitary gate folding and unitary circuit folding respectivly. Default is 'gate'.
        
    Returns:
        Folded quantum circuit
    """
    if scale_factor < 1 or int(scale_factor) % 2 == 0:
        warnings.warn(f"Scale factor {scale_factor} adjusted to nearest odd integer ≥ 1")
        scale_factor = max(1, 2 * int(np.ceil((scale_factor - 1) / 2)) + 1)
    
    scale_int = int(scale_factor)
    
    if scale_int == 1:
        copied_circuit = circuit.copy()
        copied_circuit.remove_final_measurements()
        return copied_circuit
    
    # Remove measurements for folding
    circuit_no_measure = circuit.copy()
    circuit_no_measure.remove_final_measurements()
    
    number_of_folds = (scale_int - 1) // 2
    
    # Create folded circuit
    folded_circuit = QuantumCircuit(circuit.num_qubits)

    if folding_method=='gate':
        # Get gates to fold (excluding barriers)
        gates_to_fold = []
        for instruction in circuit_no_measure.data:
            if instruction.operation.name not in ['barrier']:
                gates_to_fold.append(instruction)
    
        for instruction in gates_to_fold:
            folded_circuit.append(instruction.operation, instruction.qubits)
            
            for _ in range(number_of_folds):
                folded_circuit.append(instruction.operation, instruction.qubits)
                folded_circuit.append(instruction.operation.inverse(), instruction.qubits)

    elif folding_method=='circuit':
        # Apply original circuit
        folded_circuit.compose(circuit_no_measure, inplace=True)
        
        # Apply folding pairs (circuit + inverse circuit)
        for _ in range(number_of_folds):
            folded_circuit.compose(circuit_no_measure, inplace=True)
            folded_circuit.compose(circuit_no_measure.inverse(), inplace=True)
    else:
        raise ValueError("Unknown folding method. Possible values are 'gate' and 'circuit'.")
    
    folded_circuit.metadata = {'scale_factor': scale_factor, 'folding_method': folding_method}
    
    return folded_circuit

def zne_exponential_mitigation(
    data: Union[List[List[float]], np.ndarray],
    factors: Union[List[float], np.ndarray] = None):
    """
    Perform Zero-Noise Extrapolation (ZNE) using an exponential fit A + B * C^x.
    
    Parameters:
    -----------
    data : array-like of shape (N, 4)
        Observed expectation values for N observables at 4 different noise scaling factors.
    factors : array-like of length 4, optional
        Noise scaling factors used to generate the data. Defaults to [1, 3, 5, 7].
        
    Returns:
    --------
    mitigated_values : array of float
        Extrapolated zero-noise values (size N). Failed fits return NaN.
    """
    if factors is None:
        factors = np.array([1.0, 3.0, 5.0, 7.0])
    else:
        factors = np.asarray(factors, dtype=float)
        
    data = np.asarray(data, dtype=float)
    if data.shape[1] != len(factors):
        raise ValueError(f"Expected {len(factors)} columns in data, got {data.shape[1]}.")
        
    # Exponential model: f(x) = A + B * C^x
    def exp_model(x, A, B, C):
        return A + B * (C ** x)
        
    N = data.shape[0]
    mitigated = np.full(N, np.nan)
    
    for i in range(N):
        y = data[i, :]
        if np.any(np.isnan(y)):
            continue
            
        try:
            # Initial guesses: 
            # A ~ average observable value (noise floor)
            # B ~ offset from first point
            # C ~ 1.0 (neutral exponential factor)
            p0 = [np.mean(y), y[0] - np.mean(y), 1.0]
            
            # Constrain C to be positive to avoid complex numbers during optimization
            lower_bounds = [-np.inf, -np.inf, 1e-6]
            upper_bounds = [np.inf, np.inf, np.inf]
            
            popt, _ = curve_fit(
                exp_model,
                factors,
                y,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=2000
            )
            
            A, B, _ = popt
            # Zero-noise limit corresponds to x = 0 => f(0) = A + B*C^0 = A + B
            mitigated[i] = A + B
            
        except Exception:
            # If the optimizer fails to converge, leave as NaN
            pass
            
    return np.array(mitigated)

def zne_linear_mitigation(
    data: Union[List[List[float]], np.ndarray],
    factors: Union[List[float], np.ndarray] = None):
    """
    Perform Zero-Noise Extrapolation (ZNE) using a linear fit A + B * x.
    
    Parameters:
    -----------
    data : array-like of shape (N, 4)
        Observed expectation values for N observables at 4 different noise scaling factors.
    factors : array-like of length 4, optional
        Noise scaling factors used to generate the data. Defaults to [1, 3, 5, 7].
        
    Returns:
    --------
    mitigated_values : array of float
        Extrapolated zero-noise values (size N). Failed fits return NaN.
    """
    if factors is None:
        factors = np.array([1.0, 3.0, 5.0, 7.0])
    else:
        factors = np.asarray(factors, dtype=float)
    
    def linear_extrapolation(x: np.ndarray, y: np.ndarray) -> float:
        """Linear extrapolation to zero."""
        if len(x) < 2:
            return y[0]

        coeffs = np.polyfit(x, y, 1)
        return coeffs[1]  # y-intercept
    
    x = factors
    return np.array([linear_extrapolation(x, data_point) for data_point in data])