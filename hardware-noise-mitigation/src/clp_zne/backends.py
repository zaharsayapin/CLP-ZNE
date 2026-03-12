import numpy as np
import warnings
import datetime
from qiskit.providers.backend import BackendV2
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import RZGate, SXGate, XGate, CZGate, Measure, IGate
from qiskit.providers import QubitProperties
# V1 Models required for configuration() and properties()
from qiskit.providers.models import QasmBackendConfiguration, BackendProperties, Nduv
from qiskit.providers.models.backendproperties import Gate

class NClusterBackend(BackendV2):
    def __init__(self, target, num_qubits, cluster_size, num_clusters):
        super().__init__(name=f"{num_clusters}ClusterBackend")
        self._target = target
        self._num_qubits = num_qubits
        self._cluster_size = cluster_size
        self._num_clusters = num_clusters

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None
    
    @property
    def cluster_size(self):
        return self._cluster_size
    
    @property
    def num_qubits(self):
        return self._num_qubits
    
    @property
    def num_clusters(self):
        return self._num_clusters

    def run(self, run_input, **options):
        raise NotImplementedError("This is a synthetic backend for transpilation only.")

    @classmethod
    def _default_options(cls):
        return None
    
    # --------------------------------------------------------------------------
    # BackendV1 Compatibility Methods (Required for noise_model_from_backend)
    # --------------------------------------------------------------------------

    def configuration(self):
        """
        Returns a BackendConfiguration object compatible with Qiskit Aer's 
        NoiseModel.from_backend() legacy expectations.
        """
        # Extract basis gates from Target
        basis_gates = list(self.target.operation_names)
        
        # Extract coupling map from Target
        coupling_map = []
        cmap_obj = self.target.build_coupling_map()
        if cmap_obj is not None:
            coupling_map = list(cmap_obj.get_edges())

        return QasmBackendConfiguration(
            backend_name=self.name,
            backend_version="1.0.0",
            n_qubits=self.num_qubits,
            basis_gates=basis_gates,
            simulator=True,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=int(1e6),
            coupling_map=coupling_map,
            # Required fields for QasmBackendConfiguration
            gates=[], 
            n_registers=1,
        )


    def properties(self):
        """
        Returns a BackendProperties object mapping Target data to V1 properties.
        This allows noise_model_from_backend to extract T1, T2, gate_error, and gate_length.
        """
        qubit_props_list = []
        gate_props_list = []
        now = datetime.datetime.now(datetime.timezone.utc)

        # 1. Construct Qubit Properties (T1, T2, Frequency, Readout)
        for qp in self.target.qubit_properties:
            nduvs = [
                Nduv(date=now, name='T1', value=qp.t1 if qp.t1 is not None else 0.0, unit='s'),
                Nduv(date=now, name='T2', value=qp.t2 if qp.t2 is not None else 0.0, unit='s'),
                Nduv(date=now, name='frequency', value=5.0, unit='GHz'),
            ]
            
            qubit_props_list.append(nduvs)

        # 2. Construct Gate Properties (Error, Duration)
        for op_name, qubit_map in self.target.items():
            # Skip measurement for gate properties
            if op_name == 'measure':
                continue
            
            for qubits, props in qubit_map.items():
                if props is None:
                    continue
                
                params = []
                # Map InstructionProperties to Nduv parameters expected by noise_model_from_backend
                if props.error is not None:
                    params.append(Nduv(date=now, name='gate_error', value=props.error, unit=''))
                
                if props.duration is not None:
                    # Convert seconds to nanoseconds for better compatibility
                    # Qiskit expects time units with SI prefixes
                    duration_ns = props.duration * 1e9  # Convert s to ns
                    params.append(Nduv(date=now, name='gate_length', value=duration_ns, unit='ns'))

                gate = Gate(
                    gate=op_name,
                    qubits=list(qubits),
                    date=now,
                    parameters=params,
                    name=op_name
                )
                gate_props_list.append(gate)

        return BackendProperties(
            backend_name=self.name,
            backend_version="1.0.0",
            last_update_date=now,
            qubits=qubit_props_list,
            gates=gate_props_list,
            general=[]
        )

    def qubit_properties(self, qubit: int | list[int]):
        """
        Return QubitProperties for a given qubit.
        
        Parameters:
            qubit (int | list[int]): The qubit to get the QubitProperties object for.
                This can be a single integer for 1 qubit or a list of qubits and a 
                list of QubitProperties objects will be returned in the same order.
        
        Returns:
            QubitProperties | list[QubitProperties]: The QubitProperties object for 
                the specified qubit. If a list of qubits is provided a list will be 
                returned. If properties are missing for a qubit it will return None for that qubit.
        """
        
        if isinstance(qubit, int):
            # Single qubit: return QubitProperties object or None
            if 0 <= qubit < self._num_qubits:
                return self._target.qubit_properties[qubit]
            else:
                return None
        elif isinstance(qubit, list):
            # List of qubits: return list of QubitProperties objects
            return [
                self._target.qubit_properties[q] if 0 <= q < self._num_qubits 
                else None 
                for q in qubit
            ]
        else:
            raise TypeError(f"Expected int or list[int] for qubit, got {type(qubit)}")
    
    @classmethod
    def from_backend(cls, cluster_size, source_backend, list_of_index_groups, seed=None):
        params = NClusterBackend.get_parameters_for_initialization(source_backend, list_of_index_groups)
        return cls.from_parameters(cluster_size, **params, seed=seed)

    @classmethod
    def from_parameters(
        cls, cluster_size, t1_stats, t2_stats, sq_err_stats, tq_err_stats,
        sq_gate_time, tq_gate_time, meas_time=1e-6, meas_err=0, seed=None
    ):
        # Initialize local random generator for reproducibility
        rng = np.random.default_rng(seed)

        num_clusters = len(t1_stats)
        num_qubits = cluster_size * num_clusters
        target = Target(num_qubits=num_qubits)
        
        # --- 1. Qubit Properties ---
        qubit_props = []
        for c_idx in range(num_clusters):
            m_t1, s_t1 = t1_stats[c_idx]
            m_t2, s_t2 = t2_stats[c_idx]
            for i_in_c in range(cluster_size):
                qubit_idx = c_idx * cluster_size + i_in_c
                
                # Sample and enforce non-negative values
                t1 = max(0, rng.normal(m_t1, s_t1))
                t2 = max(0, min(rng.normal(m_t2, s_t2), 2 * t1))
                
                # --- ZERO VALUE WARNINGS ---
                if t1 == 0:
                    warnings.warn(f"Qubit {qubit_idx} has T1 = 0. Coherence is non-existent.", UserWarning)
                if t2 == 0:
                    warnings.warn(f"Qubit {qubit_idx} has T2 = 0. Phase will decohere instantly.", UserWarning)
                
                qubit_props.append(QubitProperties(t1=t1, t2=t2))
        
        target.qubit_properties = qubit_props

        # --- 2. Prepare Instruction Property Dictionaries ---
        x_props, sx_props, rz_props, id_props, meas_props = {}, {}, {}, {}, {}
        
        for c_idx in range(num_clusters):
            m_err, s_err = sq_err_stats[c_idx]
            offset = c_idx * cluster_size
            
            for i in range(offset, offset + cluster_size):
                err = max(0, rng.normal(m_err, s_err))
                phys_p = InstructionProperties(duration=sq_gate_time, error=err)
                virt_p = InstructionProperties(duration=0, error=0)
                m_p = InstructionProperties(duration=meas_time, error=meas_err)
                
                x_props[(i,)] = phys_p
                sx_props[(i,)] = phys_p
                rz_props[(i,)] = virt_p
                id_props[(i,)] = virt_p
                meas_props[(i,)] = m_p

        target.add_instruction(XGate(), x_props)
        target.add_instruction(SXGate(), sx_props)
        target.add_instruction(RZGate(0), rz_props)
        target.add_instruction(IGate(), id_props)
        target.add_instruction(Measure(), meas_props)

        # --- 3. CZ Gates (Intra-cluster All-to-All) ---
        cz_props = {}
        for c_idx in range(num_clusters):
            mean_err, std_err = tq_err_stats[c_idx]
            offset = c_idx * cluster_size
            indices = range(offset, offset + cluster_size)
            
            for i in indices:
                for j in indices:
                    if i != j:
                        err = max(0, rng.normal(mean_err, std_err))
                        cz_props[(i, j)] = InstructionProperties(duration=tq_gate_time, error=err)

        target.add_instruction(CZGate(), cz_props)
        return cls(target, num_qubits, cluster_size, num_clusters)

    @staticmethod
    def get_parameters_for_initialization(source_backend, list_of_index_groups):
        target = source_backend.target
        
        all_stats = []
        for indices in list_of_index_groups:
            t1s = [source_backend.qubit_properties(i).t1 for i in indices]
            t2s = [source_backend.qubit_properties(i).t2 for i in indices]
            sq_e = [target['sx'][(i,)].error for i in indices if (i,) in target['sx']]
            sq_t = [target['sx'][(i,)].duration for i in indices if (i,) in target['sx']]
            
            tq_e, tq_t = [], []
            op_name = 'cz' if 'cz' in target else 'cx'
            for i in indices:
                for j in indices:
                    if (i, j) in target[op_name]:
                        tq_e.append(target[op_name][(i, j)].error)
                        tq_t.append(target[op_name][(i, j)].duration)
            
            all_stats.append({
                "t1": (np.mean(t1s), np.std(t1s)),
                "t2": (np.mean(t2s), np.std(t2s)),
                "sq_e": (np.mean(sq_e), np.std(sq_e)),
                "tq_e": (np.mean(tq_e), np.std(tq_e)),
                "sq_t": np.mean(sq_t),
                "tq_t": np.mean(tq_t)
            })

        return {
            "t1_stats": [s["t1"] for s in all_stats],
            "t2_stats": [s["t2"] for s in all_stats],
            "sq_err_stats": [s["sq_e"] for s in all_stats],
            "tq_err_stats": [s["tq_e"] for s in all_stats],
            "sq_gate_time": np.mean([s["sq_t"] for s in all_stats]),
            "tq_gate_time": np.mean([s["tq_t"] for s in all_stats]),
        }