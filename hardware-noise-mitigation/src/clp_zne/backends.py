import numpy as np
from qiskit.providers.backend import BackendV2
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import RZGate, SXGate, XGate, CZGate, Measure
from qiskit.providers.models import QubitProperties

class TwinClusterBackend(BackendV2):
    def __init__(self, cluster_size, t1_c1, t1_c2, t2_c1, t2_c2, 
        sq_err_c1, sq_err_c2, tq_err_c1, tq_err_c2,
        sq_gate_time, tq_gate_time, measurement_err=0, measurement_time=1e-6
    ):
        super().__init__(name="TwinClusterBackend")

        num_qubits = cluster_size * 2
        target = Target(num_qubits=num_qubits)
        
        # --- 1. Sampling Parameters ---
        qubit_props = []
        for i in range(num_qubits):
            # Select stats based on group
            m_t1, s_t1 = t1_c1 if i < cluster_size else t1_c2
            m_t2, s_t2 = t2_c1 if i < cluster_size else t2_c2
            
            t1 = np.random.normal(m_t1, s_t1)
            # Physical constraint: T2 <= 2T1
            t2 = min(np.random.normal(m_t2, s_t2), 2 * t1)
            
            qubit_props.append(QubitProperties(t1=t1, t2=t2))

        # Assign qubit properties to target
        target.qubit_properties = qubit_props

        # --- 2. Defining Instructions ---
        # Single Qubit Gates
        for i in range(num_qubits):
            m_err, s_err = sq_err_c1 if i < cluster_size else sq_err_c2
            err = max(0, np.random.normal(m_err, s_err))
            
            sq_props = InstructionProperties(duration=sq_gate_time, error=err)
            
            # Adding basis gates from FakeTorino style
            target.add_instruction(XGate(), {(i,): sq_props})
            target.add_instruction(SXGate(), {(i,): sq_props})
            target.add_instruction(RZGate(), {(i,): InstructionProperties(duration=0, error=0)})
            target.add_instruction(Measure(), {(i,): InstructionProperties(duration=measurement_time, error=measurement_err)})

        # Two Qubit Gates (CZ) - All-to-all within groups
        def add_cz_group(indices, m_tq, s_tq):
            for i in indices:
                for j in indices:
                    if i != j:
                        err = max(0, np.random.normal(m_tq, s_tq))
                        tq_props = InstructionProperties(duration=tq_gate_time, error=err)
                        target.add_instruction(CZGate(), {(i, j): tq_props})

        add_cz_group(range(cluster_size), tq_err_c1[0], tq_err_c1[1])
        add_cz_group(range(cluster_size, num_qubits), tq_err_c2[0], tq_err_c2[1])

        self._target = target
        self._num_qubits = num_qubits
    
    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        return None
    
    def get_parameters_for_TwinClusterBackend(source_backend, g1_indices, g2_indices):
        target = source_backend.target
        
        def extract_stats(indices):
            t1s = [source_backend.qubit_properties(i).t1 for i in indices]
            t2s = [source_backend.qubit_properties(i).t2 for i in indices]
            
            # Pull errors and durations for 'sx' as the representative SQ gate
            sq_errors = [target['sx'][(i,)].error for i in indices if (i,) in target['sx']]
            sq_times = [target['sx'][(i,)].duration for i in indices if (i,) in target['sx']]

            # Pull errors for 'cx' or 'cz' pairs strictly within this group
            tq_errors, tq_times = [], []
            op_name = 'cz' if 'cz' in target else 'cx'
            for i in indices:
                for j in indices:
                    if (i, j) in target[op_name]:
                        tq_errors.append(target[op_name][(i, j)].error)
                        tq_times.append(target[op_name][(i, j)].duration)
            return (
                (np.mean(t1s), np.std(t1s)),
                (np.mean(t2s), np.std(t2s)),
                (np.mean(sq_errors), np.std(sq_errors)),
                (np.mean(tq_errors), np.std(tq_errors)),
                (np.mean(sq_times), np.std(sq_times))
                (np.mean(tq_times), np.std(tq_times)),
            )

        stats1 = extract_stats(g1_indices)
        stats2 = extract_stats(g2_indices)
        
        return {
            "t1_c1": stats1[0], "t2_c1": stats1[1], "sq_err_c1": stats1[2], "tq_err_c1": stats1[3],
            "t1_c2": stats2[0], "t2_c2": stats2[1], "sq_err_c2": stats2[2], "tq_err_c2": stats2[3],
            "sq_gate_time": (stats1[4][0]+stats2[4][0])/2, 
            "tq_gate_time": (stats1[5][0]+stats2[5][0])/2,
        }