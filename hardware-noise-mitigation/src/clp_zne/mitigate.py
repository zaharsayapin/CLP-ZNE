from dataclasses import dataclass
import warnings

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SuperOp, average_gate_fidelity
from qiskit_aer.noise import thermal_relaxation_error

from .utils import (
    transpile_to_layouts,
    cyclic_permutations,
    compute_evals,
    linear_extrapolation
)
from .noise import noise_model_from_backend

@dataclass
class ErrorProfile:
    """Container for multi-parameter error breakdown of a circuit."""
    d1: float = 0.0      # Depol 1q
    d2: float = 0.0      # Depol 2q
    t1_1: float = 0.0    # T1-related infidelity 1q
    tp_1: float = 0.0    # T-phi-related infidelity 1q
    t1_2: float = 0.0    # T1-related infidelity 2q
    tp_2: float = 0.0    # T-phi-related infidelity 2q
    therm1: float = 0.0  # Composite thermal infidelity 1q
    therm2: float = 0.0  # Composite thermal infidelity 2q
    total1: float = 0.0  # Total gate_error 1q
    total2: float = 0.0  # Total gate_error 2q

    def get_features(self, option: int) -> np.ndarray:
        """Projects the 10 parameters into N features based on ZNE option."""
        mapping = {
            1: [self.total2],
            2: [self.d2, self.therm2],
            3: [self.d2, self.t1_2, self.tp_2],
            4: [self.d2, self.t1_2, self.tp_2, self.d1],
            5: [self.d2, self.t1_2, self.tp_2, self.d1, self.therm1],
            6: [self.d2, self.t1_2, self.tp_2, self.d1, self.t1_1, self.tp_1]
        }
        return np.array(mapping.get(option, [self.total2]))

# This function populates the `ErrorProfile` by analyzing every gate in the transpiled circuit.
def calculate_circuit_error_profile(backend, tcirc, noise_model, therm_noise_multiplier=1):
    profile = ErrorProfile()
    
    for inst in tcirc.data:
        op = inst.operation
        if op is None or op.name in [None, 'rz', 'barrier', 'measure']:
            continue
            
        qubits = tuple(tcirc.qubits.index(q) for q in inst.qubits)
        n_q = len(qubits)
        name = op.name
        
        # 1. Fetch Backend reported values
        gate_err = backend.target[name][qubits].error
        if gate_err is None:
            raise ValueError(f"Not found error info for {name} on qubits {qubits}")
        
        gate_time = backend.target[name][qubits].duration
        if gate_time is None:
            raise ValueError(f"Not found gate duration info for {name} on qubits {qubits}")
        
        # 2. Calculate Thermal Components
        therm_ops = []
        t1_sum, tphi_sum = 0.0, 0.0
        
        for i, q in enumerate(qubits):
            t1 = backend.qubit_properties(q).t1
            t2 = backend.qubit_properties(q).t2

            if t1 is None or t2 is None:
                raise ValueError(f"Not found T1/T2 times for qubit {q}")
            
            t2 = min(t2, 2*t1)
            t1 /= therm_noise_multiplier
            t2 /= therm_noise_multiplier

            therm_ops.append(thermal_relaxation_error(t1, t2, gate_time))
            
            if n_q == 1:
                # Avg. fidelity of single qubit T1-noise: F_t1 = (3 + exp(-t/T1) + 2*exp(-t/2T1))/6
                t1_inf = 1 - (3 + np.exp(-gate_time/t1) + 2*np.exp(-gate_time/(2*t1)))/6

                # Avg. fidelity of single qubit T_phi-noise: F_tp = (2 + exp(-t/Tp))/3
                tp_inv = 1/t2 - 1/(2*t1)
                tp_inf = 1 - (2 + np.exp(-gate_time * tp_inv))/3
            elif n_q == 2:
                # Avg. fidelity of two qubit T1-noise, where it acts only on one of the qubits: F_t1 = (2 + exp(-t/T1) + 2*exp(-t/2T1))/5
                t1_inf = 1 - (2 + np.exp(-gate_time/t1) + 2*np.exp(-gate_time/(2*t1)))/5

                # Avg. fidelity of two qubit T_phi-noise, where it acts only on one of the qubits: F_tp = (3 + 2*exp(-t/2T1))/5
                tp_inv = 1/t2 - 1/(2*t1)
                tp_inf = 1 - (3 + 2*np.exp(-gate_time * tp_inv))/5

            t1_sum += t1_inf
            tphi_sum += tp_inf

        # Composite Thermal Error
        therm_joint = therm_ops[0]
        for e in therm_ops[1:]: therm_joint = therm_joint.tensor(e)
        f_th = average_gate_fidelity(SuperOp(therm_joint))
        inf_th = max(0.0, 1.0 - f_th)

        # 3. Calculate depolarizing residual error
        # Assume the noise structure: depol ∘ thermal,
        # where thermal noise being applied first
        d = 2**n_q
        inf_depol = max(0, (d-1)/d*(f_th-(1-gate_err))/(f_th-1/d))

        try:
            error_obj = noise_model._local_quantum_errors[name][qubits]
        except:
            error_obj = noise_model._local_quantum_errors[name][qubits[::-1]]
        total_gate_error = 1 - average_gate_fidelity(error_obj)

        # 4. Update Profile
        if n_q == 1:
            profile.total1 += total_gate_error
            profile.d1 += inf_depol
            profile.t1_1 += t1_sum
            profile.tp_1 += tphi_sum
            profile.therm1 += inf_th
        elif n_q == 2:
            profile.total2 += total_gate_error
            profile.d2 += inf_depol
            profile.t1_2 += t1_sum
            profile.tp_2 += tphi_sum
            profile.therm2 += inf_th
            
    return profile

def compute_error_sum(noise_model, circuit, cycle, connection_type):
    def minimal_number_of_neighbours_between(arr, val1, val2):
        # Check if both values exist in the list
        if val1 not in arr:
            raise ValueError(f"Value '{val1}' not found in the provided list.")
        if val2 not in arr:
            raise ValueError(f"Value '{val2}' not found in the provided list.")
        
        # Check if the values are the same
        if val1 == val2:
            raise ValueError("Both values are the same; distance cannot be calculated between identical elements.")

        # Get the indices of the two numbers
        idx1 = arr.index(val1)
        idx2 = arr.index(val2)
        
        n = len(arr)
        
        # Calculate the absolute jump between indices
        d = abs(idx1 - idx2)
        
        # The shortest "jump" distance in a cycle is min(d, n - d)
        shortest_jump = min(d, n - d)

        return shortest_jump - 1
    
    qubits_satisfy_connection_type = lambda qubits, cycle, connection_type: minimal_number_of_neighbours_between(cycle, *qubits) == connection_type

    error_sum = 0
    for instruction in circuit.data:
        gate = instruction.operation
        gate_name = gate.name
        qubits = tuple(circuit.find_bit(qubit)[0] for qubit in instruction.qubits)

        if gate_name=='cz':
            try:
                error_obj = noise_model._local_quantum_errors[gate_name][qubits]
            except:
                error_obj = noise_model._local_quantum_errors[gate_name][qubits[::-1]]
            total_gate_error = 1 - average_gate_fidelity(error_obj)

            if qubits_satisfy_connection_type(qubits, cycle, connection_type):
                error_sum += total_gate_error
    return error_sum

def get_error_matrix(circuit, layout_cycles, backend, num_params=1, therm_noise_multiplier=1):
    noise_model = noise_model_from_backend(
        backend,
        add_readout=False,
        add_gate_errors= True,
        thermal_relaxation= True,
        therm_error_multiplier=therm_noise_multiplier,
        warnings= False,
    )
    # 1. Prepare Layouts and Transpile
    cyclically_permuted_layouts = []
    for cycle in layout_cycles:
        cyclically_permuted_layouts.extend(cyclic_permutations(cycle))
    
    transpiled_circs = transpile_to_layouts(circuit, cyclically_permuted_layouts, backend.target)
    
    # 2. Build Feature Matrix X
    print("Calculating error profiles...")
    profiles = [calculate_circuit_error_profile(backend, c, noise_model, therm_noise_multiplier) for c in transpiled_circs]
    
    # Group profiles by layout groups to match the averaging in CLP
    group_size = len(cyclically_permuted_layouts) // len(layout_cycles)
    
    X_list = []
    for i in range(0, len(profiles), group_size):
        group = profiles[i : i + group_size]
        # Average the features across the cyclic permutations
        avg_features = np.mean([p.get_features(num_params) for p in group], axis=0)
        X_list.append(avg_features)
    
    X = np.array(X_list) # Shape (len(layout_cycles), n_features)
    return X

def clp_zne_mitigate_1d_topology_circuit(circuit, observables, layout_cycles, backend, num_params=1,
                                         therm_noise_multiplier=1):
    """
    ZNE Mitigation using Cyclic Layout Permutations.
    
    Args:
        num_params: 1 (Total 2Q) to 6 (Full breakdown).
    """
    n_qubits = circuit.num_qubits
    
    # 1. Prepare Layouts and Transpile
    cyclically_permuted_layouts = []
    for cycle in layout_cycles:
        cyclically_permuted_layouts.extend(cyclic_permutations(cycle))
    
    transpiled_circs = transpile_to_layouts(circuit, cyclically_permuted_layouts, backend.target)
    
    # Build noise model
    noise_model = noise_model_from_backend(
        backend,
        add_readout=False,
        add_gate_errors= True,
        thermal_relaxation= True,
        therm_error_multiplier=therm_noise_multiplier,
        warnings= False,
    )

    # 2. Build Feature Matrix X
    print("Calculating error profiles...")
    profiles = [calculate_circuit_error_profile(backend, c, noise_model, therm_noise_multiplier) for c in transpiled_circs]
    
    # Group profiles by layout groups to match the averaging in CLP
    group_size = len(cyclically_permuted_layouts) // len(layout_cycles)
    
    X_list = []
    for i in range(0, len(profiles), group_size):
        group = profiles[i : i + group_size]
        # Average the features across the cyclic permutations
        avg_features = np.mean([p.get_features(num_params) for p in group], axis=0)
        X_list.append(avg_features)
    
    X = np.array(X_list) # Shape (len(layout_cycles), n_features)
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

    # 3. Run Noisy Simulations
    print("Running simulations...")
    evals_noisy = compute_evals(
        transpiled_circs, 
        layouts=np.array(cyclically_permuted_layouts)[:, :n_qubits],
        observables=observables, 
        noise_model=noise_model
    )
    
    # 4. Perform Multi-Parameter Regression
    evals_mitigated = []
    for obs_idx in range(len(observables)):
        # Average noisy results across permutations
        y_data = evals_noisy[obs_idx]
        y = y_data.reshape((len(layout_cycles), -1)).mean(axis=1)
        
        coeffs, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        
        # The first coefficient is the intercept (noise -> 0 limit)
        evals_mitigated.append(coeffs[0])
        
    return evals_mitigated, evals_noisy, X

def reshape(data, rows, cols):
    if len(data) != rows * cols:
        raise ValueError("Total elements must match the new shape.")
    return [data[i * cols : (i + 1) * cols] for i in range(rows)]

def clp_zne_mitigate_general_topology_circuit(circuit, observables, layout_cycles, backend, therm_noise_multiplier=1):
    num_qubits = circuit.num_qubits
    num_connection_types = num_qubits // 2
    target = backend.target

    # Build noise model
    noise_model = noise_model_from_backend(
        backend,
        add_readout=False,
        add_gate_errors= True,
        thermal_relaxation= True,
        therm_error_multiplier=therm_noise_multiplier,
    )

    # Generate cyclic layout permutations (CLP)
    cyclically_permuted_layouts = []
    for cycle in layout_cycles:
        cyclically_permuted_layouts.extend(cyclic_permutations(cycle))

    # Create passmanagers for transpiling
    transpiled_circuits = transpile_to_layouts(circuit, cyclically_permuted_layouts, target,
                                                add_measurements=False, dynamical_decoupling=False)
    transpiled_circuits_reshaped = reshape(transpiled_circuits, len(layout_cycles), num_qubits)
    
    # Compute error sums
    error_mtx = np.zeros((len(layout_cycles), num_connection_types))
    for cycle_idx, cycle in enumerate(layout_cycles):
        for connection_type in range(num_connection_types):
            errors = [compute_error_sum(noise_model, tcirc, cycle, connection_type) for tcirc in transpiled_circuits_reshaped[cycle_idx]]
            average_error = np.mean(errors)
            error_mtx[cycle_idx, connection_type] = average_error
    print(error_mtx)
    
    # Run with noise
    print("Running density matrix simulations")
    evals_noisy = compute_evals(transpiled_circuits, layouts=np.array(cyclically_permuted_layouts)[:, :num_qubits],
                                    observables=observables, noise_model=noise_model)
    # Iterate over observables
    error_sums = []
    evals_mitigated = []
    for i, observable in enumerate(observables):
        # Perform averaging
        X = error_mtx
        y_data = evals_noisy[i]
        y = y_data.reshape((len(layout_cycles), -1)).mean(axis=1).reshape((len(layout_cycles), 1))

        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

        coeffs = X_with_intercept.T @ np.linalg.inv(X_with_intercept @ X_with_intercept.T) @ y
        
        eval_mitigated = coeffs[0, 0]

        evals_mitigated.append(eval_mitigated)
        error_sums.append(X)

    return evals_mitigated, evals_noisy, error_sums

def zne_mitigate(circuit, observables, layout, backend, therm_noise_multiplier=1, folding_method='gate'):
    """
    Implements Digital Zero-Noise Extrapolation.
    
    :param circuit: quantum circuit.
    :param observables: list of observables.
    :param layout: qubit layout.
    :param backend: backend to run the circuit on.
    :param noise_model: noise model used in simulation.
    :param folding_method: method to use for noise amplification. Possible values are 'gate' and 'circuit' to perform unitary gate folding and unitary circuit folding respectivly. By default is 'gate'.
    """
    n_qubits = circuit.num_qubits
    target = backend.target

    transpiled_circuit = transpile_to_layouts(circuit, [layout], target, add_measurements=False, dynamical_decoupling=False)[0]
    scaled_circuits = []
    scale_factors = [1, 3, 5, 7]

    for scale_factor in scale_factors:
        scaled_circuits.append(fold_circuit(transpiled_circuit, scale_factor, folding_method=folding_method))

    # Run with noise
    # Build noise model
    noise_model = noise_model_from_backend(
        backend,
        add_readout=False,
        add_gate_errors= True,
        thermal_relaxation= True,
        therm_error_multiplier=therm_noise_multiplier,
        warnings= False,
    )
    print("Running density matrix simulations")
    evals_noisy = compute_evals(scaled_circuits, layouts=np.array([layout]*len(scaled_circuits))[:, :n_qubits],
                                    observables=observables, noise_model=noise_model)
    
    # Iterate over observables
    evals_mitigated = []
    for noisy_values in evals_noisy:
        eval_mitigated = linear_extrapolation(x=scale_factors, y=noisy_values)
        evals_mitigated.append(eval_mitigated)

    return evals_mitigated, evals_noisy

def fold_circuit(circuit: QuantumCircuit, scale_factor: float, folding_method='gate') -> QuantumCircuit:
    """
    Fold a quantum circuit to amplify noise. Removes all end circuit measurments.
    
    Args:
        circuit: Original quantum circuit
        scale_factor: Noise amplification factor (must be odd: 1, 3, 5, ...)
        folding_method: method to use for noise amplification. Possible values are 'gate' and 'circuit' to perform unitary gate folding and unitary circuit folding respectivly. By default is 'gate'.
        
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