from qiskit import QuantumCircuit
from qiskit_aer.utils import insert_noise
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix
from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error
from qiskit.quantum_info import average_gate_fidelity, SuperOp, Operator
from .utils import build_subset_circuit, transpile_to_layouts, cyclic_permutations, linear_extrapolation
import warnings
from tqdm import tqdm
import numpy as np

def get_t1_t2(props, q):
    # try multiple APIs (works with BackendProperties v1+)
    for fn in ('t1', 'T1',):
        try:
            val = getattr(props, fn)(q)
            return val, getattr(props, 't2')(q)
        except Exception:
            pass
    # fallback to qubit_property if available
    try:
        t1 = props.qubit_property(q, 'T1')[0]
        t2 = props.qubit_property(q, 'T2')[0]
        return t1, t2
    except Exception:
        return None, None

def get_contibution_infidelities_with_single_qubit_errors(backend, tcirc, atol=1e-10, verbose=False):
    props = backend.properties()

    T1_up_infidelities = []
    T_phi_up_infidelities = []
    depol_infidelities_1 = []
    depol_infidelities_2 = []
    
    for inst in tcirc.data:
        op = inst.operation
        # skip non-gate ops
        if op is None or op.name is None or op.name=='rz':
            continue
            
        if len(inst.qubits) == 2:
            assert len(inst.qubits) == 2
            name = op.name
            # map instruction qubits -> integer indices in the transpiled circuit
            qubits = tuple(tcirc.qubits.index(q) for q in inst.qubits)
            n_q = len(qubits)
            # get backend-reported gate_error and gate_length (seconds)
            try:
                gate_err = props.gate_error(name, qubits)
            except Exception:
                gate_err = None
            try:
                gate_time = props.gate_length(name, qubits)  # seconds
            except Exception:
                gate_time = None

            # get per-qubit T1/T2 and create single-qubit thermal errors (or identity)
            therm_errors = []
            for i, q in enumerate(qubits):
                t1, t2 = get_t1_t2(props, q)
                t2 = min(t2, 2*t1)

                if i == 0:
                    T1_up_inf = 1 - (2+np.exp(-gate_time/t1)+2*np.exp(-gate_time/(2*t1)))/5
                    T_phi_inv = 1/t2 - 1/(2*t1)
                    T_phi_up_inf = 1 - (3+2*np.exp(-gate_time*T_phi_inv))/5

                    T1_up_infidelities.append(T1_up_inf)
                    T_phi_up_infidelities.append(T_phi_up_inf)

                if t1 is None or t2 is None or gate_time is None:
                    therm_errors.append(depolarizing_error(0.0, 1))
                else:
                    therm_errors.append(thermal_relaxation_error(t1, t2, gate_time))

            # build joint thermal error (tensor of single-qubit thermal errors)
            therm = therm_errors[0]
            for e in therm_errors[1:]:
                therm = therm.tensor(e)

            # average fidelity and infidelity of thermal part
            try:
                superop_therm = SuperOp(therm)
                dim = 2 ** n_q
                identity = Operator(np.identity(dim))
                f_th = average_gate_fidelity(superop_therm, identity)
            except Exception:
                f_th = 1.0
            inf_th = max(0.0, 1.0 - f_th)

            # If backend doesn't report gate_error, Qiskit.from_backend will typically
            # insert only thermal errors (no depolarizing) — we mirror that.
            if gate_err is None:
                raise ValueError("Gate_err is None")
                if verbose:
                    print(f"{name}{qubits}: gate_err=N/A, thermal_inf={inf_th:.3e}, depol_inf=0")
                continue


            # if thermal infidelity >= reported gate_error, Qiskit uses thermal only
            if inf_th >= gate_err - atol:
                depol_infidelities_2.append(0)
                if verbose:
                    print(f"{name}{qubits}: gate_err={gate_err:.3e}, thermal_inf={inf_th:.3e} (>= gate_err) -> no depol")
                continue

            # otherwise find depolarizing parameter λ so that
            # average_fidelity( depol(λ) ∘ thermal ) == 1 - gate_err
            target_f = 1.0 - gate_err

            depol_inf = 3/4*(f_th-target_f)/(f_th-1/4)
            depol_infidelities_2.append(depol_inf)      

            if verbose:
                 print(f"{name}{qubits}: gate_err={gate_err:.3e}, thermal_inf={inf_th:.3e}, depol_inf={depol_inf:.3e}")
        
        elif len(inst.qubits) == 1:
            assert len(inst.qubits) == 1
            name = op.name
            # map instruction qubits -> integer indices in the transpiled circuit
            qubits = tuple(tcirc.qubits.index(q) for q in inst.qubits)
            n_q = len(qubits)
            # get backend-reported gate_error and gate_length (seconds)
            try:
                gate_err = props.gate_error(name, qubits)
            except Exception:
                gate_err = None
            try:
                gate_time = props.gate_length(name, qubits)  # seconds
            except Exception:
                gate_time = None

            # get per-qubit T1/T2 and create single-qubit thermal errors (or identity)
            therm_errors = []
            for i, q in enumerate(qubits):
                t1, t2 = get_t1_t2(props, q)
                t2 = min(t2, 2*t1)

                if i == 0:
                    T1_up_inf = 1 - (2+np.exp(-gate_time/t1)+2*np.exp(-gate_time/(2*t1)))/5
                    T_phi_inv = 1/t2 - 1/(2*t1)
                    T_phi_up_inf = 1 - (3+2*np.exp(-gate_time*T_phi_inv))/5
                    
                    
                    #T1_up_infidelities.append(T1_up_inf)
                    #T_phi_up_infidelities.append(T_phi_up_inf)

                if t1 is None or t2 is None or gate_time is None:
                    therm_errors.append(depolarizing_error(0.0, 1))
                    raise ValueError()
                else:
                    therm_errors.append(thermal_relaxation_error(t1, t2, gate_time))

            # build joint thermal error (tensor of single-qubit thermal errors)
            therm = therm_errors[0]
            for e in therm_errors[1:]:
                therm = therm.tensor(e)

            # average fidelity and infidelity of thermal part
            try:
                superop_therm = SuperOp(therm)
                dim = 2 ** n_q
                identity = Operator(np.identity(dim))
                f_th = average_gate_fidelity(superop_therm, identity)
            except Exception:
                f_th = 1.0
            inf_th = max(0.0, 1.0 - f_th)

            # If backend doesn't report gate_error, Qiskit.from_backend will typically
            # insert only thermal errors (no depolarizing) — we mirror that.
            if gate_err is None:
                raise ValueError("Gate_err is None")
                if verbose:
                    print(f"{name}{qubits}: gate_err=N/A, thermal_inf={inf_th:.3e}, depol_inf=0")
                continue


            # if thermal infidelity >= reported gate_error, Qiskit uses thermal only
            if inf_th >= gate_err - atol:
                depol_infidelities_1.append(0)
                if verbose:
                    print(f"{name}{qubits}: gate_err={gate_err:.3e}, thermal_inf={inf_th:.3e} (>= gate_err) -> no depol")
                continue

            # otherwise find depolarizing parameter λ so that
            # average_fidelity( depol(λ) ∘ thermal ) == 1 - gate_err
            target_f = 1.0 - gate_err

            depol_inf = 1/2*(f_th-target_f)/(f_th-1/2)
            depol_infidelities_1.append(depol_inf)      

            if verbose:
                 print(f"{name}{qubits}: gate_err={gate_err:.3e}, thermal_inf={inf_th:.3e}, depol_inf={depol_inf:.3e}")
            
    return T1_up_infidelities, T_phi_up_infidelities, depol_infidelities_1, depol_infidelities_2


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
        for observable in tqdm(observables):
            evs.append(density_matrix.expectation_value(observable).real)
            #observable_matrix = observable.to_matrix()
            #evs.append(np.trace(density_matrix @ observable_matrix).real)

        evals_noisy.append(evs)

    return np.array(evals_noisy).T

def CLP_ZNE_mitigate(abstract_cirquit, observables, layouts, backend, noise_model, transpiled_circuits=None):
    # Check that exactly 5 layouts are provided
    assert len(layouts) == 5
    
    n_qubits = abstract_cirquit.num_qubits
    target = backend.target

    # Generate cyclic layout permutations (CLP)
    layout_cycles = []
    for cycle in layouts:
        layout_cycles.extend(cyclic_permutations(cycle))
        
    if transpiled_circuits is None:
        # Create passmanagers for transpiling
        transpiled_circuits = transpile_to_layouts(abstract_cirquit, 
                                                   layout_cycles, target, add_measurements=False, dynamical_decoupling=False)

    # Compute error sums
    q1_avg = []
    q2_avg = []
    q3_avg = []
    q4_avg = []
    for i, circuit in enumerate(transpiled_circuits):
        if i % 12==0:
            q1, q2, q3, q4 = get_contibution_infidelities_with_single_qubit_errors(backend, circuit, verbose=False)
            if np.sum(q1) >= 1 or np.sum(q2) >= 1 or np.sum(q3) >= 1 or np.sum(q4) >= 1:
                raise ValueError('Something went wrong!!!!!1111!1')
            q1_avg.append(np.sum(q1))
            q2_avg.append(np.sum(q2))
            q3_avg.append(np.sum(q3))
            q4_avg.append(np.sum(q4))
    error_mtx = np.column_stack((q1_avg, q2_avg, q3_avg, q4_avg))
    
    # Run with noise
    print("Running density matrix simulations")
    evals_noisy = compute_evals(transpiled_circuits, layouts=np.array(layout_cycles)[:, :n_qubits],
                                    observables=observables, noise_model=noise_model)
    # Iterate over observables
    error_sums = []
    evals_mitigated = []
    for i, observable in enumerate(observables):
        # Perform averaging
        X = error_mtx
        y_data = evals_noisy[i]
        y = y_data.reshape((5, -1)).mean(axis=1).reshape((5, 1))

        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

        coeffs = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        
        eval_mitigated = coeffs[0, 0]

        evals_mitigated.append(eval_mitigated)
        error_sums.append(X)

    return evals_mitigated, evals_noisy, error_sums

def ZNE_mitigate(circuit, observables, layout, backend, noise_model, folding_method='gate'):
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