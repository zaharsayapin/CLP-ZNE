################################################################
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer.noise import NoiseModel
from qiskit_aer.utils import insert_noise
from qiskit.primitives import StatevectorEstimator
import numpy as np

def cyclic_permutations(lst):
    n = len(lst)
    return [lst[i:] + lst[:i] for i in range(n)]

def compute_evals_ideal(circuit, observables):
    ideal_estimator = StatevectorEstimator()

    evals_ideal = []
    for observable in observables:
        job_ideal = ideal_estimator.run([(circuit, observable)])
        ev_ideal = job_ideal.result()[0].data.evs
        evals_ideal.append(ev_ideal)

    return evals_ideal

def keep_only_two_qubit_errors(noise_model: NoiseModel) -> NoiseModel:
    new_model = NoiseModel(basis_gates=noise_model.basis_gates)

    # Loop over all instructions that have errors
    for instr in noise_model._local_quantum_errors:
        for qubits, error in noise_model._local_quantum_errors[instr].items():
            if len(qubits) == 2:  # keep only 2-qubit errors
                new_model.add_quantum_error(error, instr, list(qubits))

    return new_model

def build_subset_circuit(original_circuit, subset_qubits, qubit_mapping):
    subset_circuit = QuantumCircuit(len(subset_qubits))
    
    for instr in original_circuit:
        # Get qubits involved in the instruction
        qubits = [original_circuit.find_bit(q)[0] for q in instr.qubits]
        
        # Check if all qubits are in the subset
        if all(q in subset_qubits for q in qubits):
            mapped_qubits = [qubit_mapping[q] for q in qubits]
            subset_circuit.append(instr.operation, mapped_qubits)
    
    return subset_circuit

def simulate_density_matrix(circuit, layout, noise_model):
    qubit_mapping = {orig: idx for idx, orig in enumerate(layout)}
    noisy_circuit = insert_noise(circuit, noise_model)
    aer_circuit = build_subset_circuit(noisy_circuit, layout, qubit_mapping)
    aer_circuit.save_state()
    simulator = AerSimulator(method='density_matrix')
    result = simulator.run(aer_circuit).result()
    density_matrix = np.asarray(result.data(0)['density_matrix'])
    return np.array(density_matrix)

def compute_evals(circuits, layouts, observables, noise_model):
    aer_circuits = []
    
    for i, circuit in enumerate(circuits):
        subset_qubits = layouts[i]
        qubit_mapping = {orig: idx for idx, orig in enumerate(subset_qubits)}
        circ = insert_noise(circuit, noise_model)
        circ = build_subset_circuit(circ, subset_qubits, qubit_mapping)
        circ.save_state()
        aer_circuits.append(circ)
    #####
    simulator = AerSimulator(method='density_matrix')
    
    evals_noisy = []
    for circuit in tqdm(aer_circuits):
        result = simulator.run(circuit).result()
        density_matrix = np.asarray(result.data(0)['density_matrix'])

        if not isinstance(density_matrix, DensityMatrix):
            density_matrix = DensityMatrix(density_matrix)

        evs = []
        for observable in observables:
            evs.append(density_matrix.expectation_value(observable).real)
            #observable_matrix = observable.to_matrix()
            #evs.append(np.trace(density_matrix @ observable_matrix).real)

        evals_noisy.append(evs)

    return np.array(evals_noisy).T

def permutation_map_of_physical_qubits(mapping_A, mapping_B, num_qubits):
    ancillas_mapping_B = sorted(list(set(range(num_qubits))-set(mapping_B)))

    permutation_map = []
    ancilla_index = 0
    for qubit_A in range(num_qubits):
        if qubit_A in mapping_A:
            virtual_qubit = mapping_A.index(qubit_A)
            qubit_B = mapping_B[virtual_qubit]
        else:
            qubit_B = ancillas_mapping_B[ancilla_index]
            ancilla_index += 1
        permutation_map.append((qubit_A, qubit_B))
        
    permutation_map = dict(permutation_map)

    return permutation_map

def transpile_to_layouts(circuit, layouts, target, optimization=3, add_measurements=False, dynamical_decoupling=False, seed_transpiler=1234):
    init_layout = layouts[0]
    
    pm = generate_preset_pass_manager(optimization, target=target, initial_layout=init_layout, seed_transpiler=seed_transpiler)
    t_circuit = pm.run(circuit)

    if dynamical_decoupling:
        t_circuit = BasisTranslator(sel, basis_gates)(dd_pm.run(t_circuit))

    transpiled_circuits = []
    for l in layouts:
        qr = QuantumRegister(target.num_qubits, 'q')
        cr = ClassicalRegister(len(init_layout), 'c')

        permutation_map = permutation_map_of_physical_qubits(init_layout, l, target.num_qubits)
        remapped_qc = QuantumCircuit(qr, cr)
    
        for instruction, qargs, cargs in t_circuit.data:
            original_physical_indices = [t_circuit.find_bit(qarg)[0] for qarg in qargs]
            new_physical_indices = [permutation_map[idx] for idx in original_physical_indices]
            new_qargs = [remapped_qc.qubits[idx] for idx in new_physical_indices]
            
            remapped_qc.append(instruction, new_qargs, cargs)

        if add_measurements:
            for virtual_qubit, physical_qubit in enumerate(l):
                remapped_qc.measure(qr[physical_qubit], cr[virtual_qubit])
    
        transpiled_circuits.append(remapped_qc)
    
    return transpiled_circuits

def reverse_qubit_order(probs: np.ndarray, n: int) -> np.ndarray:
    """
    Convert probability vector between MSB-first (leftmost qubit = q0)
    and LSB-first (rightmost qubit = q0) conventions.

    Parameters
    ----------
    probs : np.ndarray
        Length 2^n probability vector.
    n : int
        Number of qubits.

    Returns
    -------
    np.ndarray
        Probability vector with qubit order reversed.
    """
    if probs.shape[0] != 2**n:
        raise ValueError("Length of probs must be 2^n.")
    # indices 0..2^n-1
    indices = np.arange(2**n)
    # reverse bits of each index
    reversed_indices = np.array([int(f"{i:0{n}b}"[::-1], 2) for i in indices])
    return probs[reversed_indices]

#######

import numpy as np

def sample_bitstrings(probabilities, n_shots=1024, seed=None):
    """
    Sample bitstrings according to the given probability distribution.

    Args:
        probabilities (array-like): Probabilities of each bitstring (must sum to 1).
        n_shots (int): Number of samples (shots) to draw.

    Returns:
        dict: A dictionary mapping bitstring -> count.
    """
    # Ensure normalization
    probabilities = np.array(probabilities, dtype=float)
    probabilities /= probabilities.sum()
    
    # Determine number of qubits (bitstring length)
    n = int(np.log2(len(probabilities)))
    if 2**n != len(probabilities):
        raise ValueError("Length of probability vector must be a power of 2.")
    
    # Generate bitstrings in lexicographic order
    bitstrings = [format(i, f'0{n}b') for i in range(2**n)]
    
    # Sample from distribution
    np.random.seed(seed)
    samples = np.random.choice(bitstrings, size=n_shots, p=probabilities)
    
    # Count occurrences
    counts = {b: 0 for b in bitstrings}
    for s in samples:
        counts[s] += 1
    
    # Optionally remove zero-count entries
    counts = {k: v for k, v in counts.items() if v > 0}
    
    return counts

import numpy as np

def sample_bitstrings_with_readout_errors(probabilities, backend, qubit_list, n_shots=1024, spam_multiplier=1):
    """
    Sample bitstrings from a probability distribution and add realistic readout errors
    using data from backend.properties().

    Args:
        probabilities (array-like): Ideal probabilities of computational basis states.
        backend_properties: Qiskit backend.properties() object.
        n_shots (int): Number of measurement shots to simulate.

    Returns:
        dict: A dictionary mapping (possibly corrupted) bitstrings to counts.
    """
    backend_properties = backend.properties()
    
    # Normalize probability distribution
    probabilities = np.array(probabilities, dtype=float)
    probabilities /= probabilities.sum()
    
    # Determine number of qubits
    n_qubits = int(np.log2(len(probabilities)))
    if 2**n_qubits != len(probabilities):
        raise ValueError("Length of probability vector must be a power of 2.")
    
    # Build list of bitstrings in lexicographic order
    bitstrings = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    
    # --- Step 1: Draw ideal samples according to ideal probabilities ---
    samples = np.random.choice(bitstrings, size=n_shots, p=probabilities)
    
    # --- Step 2: Get per-qubit readout error probabilities ---
    prob_meas0_prep1 = []
    prob_meas1_prep0 = []
    for q in range(len(backend_properties.qubits)):
        prob_meas0_prep1.append(spam_multiplier * backend_properties.qubit_property(q, 'prob_meas0_prep1')[0])
        prob_meas1_prep0.append(spam_multiplier * backend_properties.qubit_property(q, 'prob_meas1_prep0')[0])
    
    # --- Step 3: Apply readout errors to each bit ---
    noisy_samples = []
    for s in samples:
        bits = list(s)
        for i, b in enumerate(bits):
            if b == '0':
                # True 0 measured as 1 with probability prob_meas1_prep0
                if np.random.rand() < prob_meas1_prep0[qubit_list[i]]:
                    bits[i] = '1'
            else:
                # True 1 measured as 0 with probability prob_meas0_prep1
                if np.random.rand() < prob_meas0_prep1[qubit_list[i]]:
                    bits[i] = '0'
        noisy_samples.append(''.join(bits))
    
    # --- Step 4: Count results ---
    counts = {}
    for s in noisy_samples:
        counts[s] = counts.get(s, 0) + 1
    
    return counts

from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate

def add_hadamards(circ: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of `circ` with a Hadamard gate applied to every qubit at the end."""
    new_circ = circ.copy()              # make a copy so you don’t modify the original
    for qubit in range(circ.num_qubits):
        new_circ.h(qubit)               # append an H to each qubit
    return new_circ

from qiskit.result import Counts

def to_qiskit_counts_format(counts_dict):
    """
    Convert a plain bitstring->count dictionary into the same format
    as Qiskit's pub_result.data.c.get_counts().

    Args:
        counts_dict (dict): Mapping of bitstrings to counts (e.g., {'00': 512, '01': 512})

    Returns:
        Counts: A Qiskit Counts object with bit order reversed (little-endian)
    """
    # Reverse bit order to match Qiskit's internal convention
    qiskit_counts = {k[::-1]: v for k, v in counts_dict.items()}
    
    # Wrap in a Qiskit Counts object
    return Counts(qiskit_counts)


import numpy as np
from qiskit.quantum_info import SparsePauliOp

def estimate_observable_from_counts(H: SparsePauliOp, counts_z: dict, counts_x: dict):
    """
    Estimate <H> = <H_z> + <H_x> from separate Z- and X-basis measurement counts.

    Args:
        H (SparsePauliOp): Observable composed only of Z- and X-type Pauli terms (no Y or mixed terms).
        counts_z (dict): Measurement counts in Z basis {bitstring: count}.
        counts_x (dict): Measurement counts in X basis {bitstring: count}.

    Returns:
        dict: {
            'expval_H': float,
            'stderr_H': float,
            'expval_Hz': float,
            'stderr_Hz': float,
            'expval_Hx': float,
            'stderr_Hx': float
        }
    """

    # --- Helper: expectation and variance from counts ---
    def expval_from_counts(H_part: SparsePauliOp, counts: dict):
        if len(H_part.paulis) == 0 or len(counts) == 0:
            return 0.0, 0.0, 0

        bitstrings = list(counts.keys())
        shots = sum(counts.values())
        n_qubits = len(bitstrings[0])

        # Convert bitstring to array of ±1
        outcomes = np.zeros((shots, n_qubits), dtype=int)
        idx = 0
        for bitstr, c in counts.items():
            vals = 1 - 2 * np.array(list(bitstr), dtype=int)  # 0→+1, 1→-1
            outcomes[idx:idx+c, :] = vals
            idx += c

        # Evaluate operator value per shot
        values = np.zeros(shots)
        for coeff, pauli in zip(H_part.coeffs, H_part.paulis):
            z_mask = np.array(pauli.z, dtype=bool)
            x_mask = np.array(pauli.x, dtype=bool)
            mask = np.logical_or(z_mask, x_mask)
            if np.any(mask):
                term_val = np.prod(outcomes[:, mask], axis=1)
            else:
                term_val = np.ones(shots)
            values += coeff.real * term_val

        mean = np.mean(values)
        var = np.var(values, ddof=1)
        return mean, var, shots

    # --- Separate H into H_z (only Z) and H_x (only X) ---
    H_z_terms, H_x_terms = [], []
    for p, c in zip(H.paulis, H.coeffs):
        if np.all(p.x == 0):        # only Z and I
            H_z_terms.append((p, c))
        elif np.all(p.z == 0):      # only X and I
            H_x_terms.append((p, c))
        else:
            raise ValueError(f"Term {p} contains both X and Z; cannot split cleanly.")

    H_z = (SparsePauliOp([p for p, _ in H_z_terms], [c for _, c in H_z_terms])
           if H_z_terms else SparsePauliOp(["I"*H.num_qubits], [0]))
    H_x = (SparsePauliOp([p for p, _ in H_x_terms], [c for _, c in H_x_terms])
           if H_x_terms else SparsePauliOp(["I"*H.num_qubits], [0]))

    # --- Compute statistics for each part ---
    mean_z, var_z, N_z = expval_from_counts(H_z, counts_z)
    mean_x, var_x, N_x = expval_from_counts(H_x, counts_x)

    mean_total = mean_z + mean_x
    stderr_z = np.sqrt(var_z / N_z) if N_z > 0 else 0.0
    stderr_x = np.sqrt(var_x / N_x) if N_x > 0 else 0.0
    stderr_total = np.sqrt(stderr_z**2 + stderr_x**2)

    return {
        "expval_H": mean_total,
        "stderr_H": stderr_total,
        "expval_Hz": mean_z,
        "stderr_Hz": stderr_z,
        "expval_Hx": mean_x,
        "stderr_Hx": stderr_x,
    }

def get_grid_entanglement(rows, cols):
    """Generates [control, target] pairs for an N x M grid."""
    links = []
    for i in range(rows):
        for j in range(cols):
            curr = i * cols + j
            # Horizontal link (Right)
            if j + 1 < cols:
                links.append([curr, curr + 1])
            # Vertical link (Down)
            if i + 1 < rows:
                links.append([curr, curr + cols])
    return links

def linear_extrapolation(x: np.ndarray, y: np.ndarray) -> float:
    """Linear extrapolation to zero."""
    if len(x) < 2:
        return y[0]
    
    coeffs = np.polyfit(x, y, 1)
    return coeffs[1]  # y-intercept