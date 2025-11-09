"""
Reimplementation of NoiseModel.from_backend() (qiskit-aer)
"""

from typing import Optional, Tuple, Dict, Any, List
from ..fidelity import therm_infidelity
from qiskit.quantum_info import average_gate_fidelity

import math

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)
from qiskit_aer.noise import device as device_utils

# NOTE: tests below access some private attributes of NoiseModel (e.g.
# _local_quantum_errors) because the public API for serializing/deserializing
# noise models has changed across versions. For unit-testing purposes this is
# acceptable, but production code should use the public methods.

def trim_after_first_digit(text):
    """
    Trim a string after the first encountered digit.
    """
    for i, char in enumerate(text):
        if char.isdigit():
            return text[:i]
    return text


def noise_model_from_backend(
    backend,
    add_readout: bool = True,
    add_gate_errors: bool = True,
    thermal_relaxation: bool = True,
    therm_error_multiplier: int = 1,
    warnings: bool = False,
) -> NoiseModel:
    """Construct a qiskit_aer.noise.NoiseModel from a provider backend object.
    This function follows the same high-level policy as Qiskit Aer' NoiseModel.
    It extracts:
      - basis_gates from backend.configuration().basis_gates
      - per-gate gate_error and gate_length from backend.properties().gates
      - per-qubit T1/T2 via device.thermal_relaxation_values(properties)
      - per-qubit readout error when available (tries a few common places)

    The implementation intentionally keeps a simple (and readable) structure so
    you can modify behaviour further down your development line.
    """

    # Basis gates
    basis_gates = None
    try:
        cfg = backend.configuration()
        basis_gates = getattr(cfg, "basis_gates", None)
    except Exception:
        cfg = None

    nm = NoiseModel(basis_gates=basis_gates)

    # Properties
    try:
        props = backend.properties()
    except Exception:
        props = None

    # Thermal relaxation values (T1, T2, freq) per qubit
    tvals: Optional[List[Tuple[float, float, float]]] = None
    if thermal_relaxation and props is not None:
        try:
            tvals = device_utils.thermal_relaxation_values(props)
        except Exception:
            # If device utilities fail for this backend shape, ignore thermal
            # relaxation.
            tvals = None

    # Helper to safely extract a parameter value from gate.parameters
    def _param_value(params, name):
        for p in params:
            # different providers may use either 'name' or 'parameter' attr
            p_name = getattr(p, "name", None) or getattr(p, "parameter", None)
            if p_name == name:
                return getattr(p, "value", None)
        return None

    # Parse gate errors
    if props is not None:
        for gate in getattr(props, "gates", []):
            gname = trim_after_first_digit(gate.name) # standard gate name ex. 'cz1_0' -> 'cz'
            qubits = tuple(getattr(gate, "qubits", []))

            gate_error = _param_value(getattr(gate, "parameters", []), "gate_error")
            # some backends use 'gate_length' others 'gate_time'
            gate_length = _param_value(getattr(gate, "parameters", []), "gate_length")
            if gate_length is None:
                gate_length = _param_value(getattr(gate, "parameters", []), "gate_time")
            
            n_qubits = len(qubits) if qubits else 1

            # If we don't have gate_error but thermal_relaxation is requested,
            # we still add thermal relaxation errors (matching Qiskit behaviour
            # when requested that way)
            if gate_error is None and not thermal_relaxation:
                continue
                
            therm_err = None
            dep_err = None

            if thermal_relaxation and tvals is not None and gate_length is not None:
                # For single qubit gates: thermal_relaxation_error(t1,t2,time)
                # For multi-qubit gates: Qiskit constructs single-qubit
                # thermal errors for each qubit participating in the multi-qubit
                # gate and composes them before the depolarizing error. Here we
                # will create a per-qubit thermal error and, when n_qubits>1,
                # we will create a tensor of single-qubit errors via
                # successive tensoring.
                try:
                    # Compose single-qubit thermal errors
                    therms = []
                    for q in qubits:
                        try:
                            t1, t2, _freq = tvals[q]
                            t1 /= therm_error_multiplier
                            t2 /= therm_error_multiplier
                        except Exception:
                            t1, t2 = float("inf"), float("inf")
                        # If T1/T2 missing or inf, skip thermal for this qubit
                        if not math.isfinite(t1) or not math.isfinite(t2):
                            therms.append(None)
                        else:
                            t2_clipped = min(t2, 2*t1)
                            therms.append(thermal_relaxation_error(t1, t2_clipped, gate_length))

                    # If all therms are None, no thermal composed error
                    if any(therms):
                        # If single qubit just take it
                        if n_qubits == 1:
                            therm_err = therms[0]
                        else:
                            # Tensor thermal errors into one multi-qubit error.
                            # We do this by tensoring the channel representations
                            # where available. The noise API provides a convenient
                            # tensor method for QuantumError objects through the
                            # `tensor` method (available in qiskit-aer).
                            # We conservatively build the tensor by starting from
                            # the first non-None and tensoring subsequent ones (or
                            # identity-like noop if None).
                            curr = None
                            for te in therms:
                                if te is None:
                                    # Use an identity (no-op) represented by
                                    # depolarizing_error(0,1)
                                    te_use = depolarizing_error(0.0, 1)
                                else:
                                    te_use = te
                                if curr is None:
                                    curr = te_use
                                else:
                                    try:
                                        curr = te_use.tensor(curr)
                                    except Exception:
                                        # If tensor fails, give up on building
                                        # multi-qubit thermal composition.
                                        curr = None
                                        break
                            therm_err = curr
                except Exception as e:
                    therm_err = None
                  
                
                therm_inf = 1-average_gate_fidelity(therm_err)
                
#                 T1_times = []
#                 T2_times = []
#                 for q in qubits:
#                     t1, t2, _freq = tvals[q]
#                     T1_times.append(t1)
#                     T2_times.append(min(t2, 2*t1))
#                 therm_infidelity(T1_times, T2_times, t=gate_length, qubit1=0, qubit2=1)
            
            if add_gate_errors and gate_error is not None:
                dim = 2**len(qubits)
                if thermal_relaxation and therm_inf < gate_error:
                    dep_factor = dim*(gate_error - therm_inf)/(dim*(1-therm_inf) - 1)
                    dep_factor = min(dep_factor, dim**2/(dim**2-1))
                    dep_err = depolarizing_error(dep_factor, n_qubits)
                elif not thermal_relaxation:
                    dep_factor = dim/(dim-1)*gate_error
                    dep_factor = min(max(dep_factor, 0.0), dim**2/(dim**2-1))
                    dep_err = depolarizing_error(dep_factor, n_qubits)
            if dep_err is not None and therm_err is not None:
                composed = therm_err.compose(dep_err)
                nm.add_quantum_error(composed, [gname], list(qubits), warnings=warnings)
            elif dep_err is not None:
                nm.add_quantum_error(dep_err, [gname], list(qubits), warnings=warnings)
            elif therm_err is not None:
                nm.add_quantum_error(therm_err, [gname], list(qubits), warnings=warnings)

    # Readout errors
    if props is not None and add_readout:
        # Try a handful of common places for readout calibration. Many
        # backends provide readout error in properties.as readout assignment
        # or in the general section. We try properties.readout_error if
        # present, otherwise we scan props.qubits entries for a measurement
        # assignment probability ("readout_error"/"assignment_error").
        # If none found, we skip readout.
        readouts_found = {}
        # Some providers expose a convenience call `backend.readout_errors()`
        if hasattr(backend, "get_readout_error"):
            try:
                ro = backend.get_readout_error()
                if ro is not None:
                    # user-provided object or dict
                    readouts_found = ro
            except Exception:
                pass

        # Fallback parse qubit properties
        if not readouts_found:
            try:
                qubits_props = getattr(props, "qubits", [])
                for qi, qps in enumerate(qubits_props):
                    # qps is a list of NamedData objects for this qubit
                    for entry in qps:
                        name = getattr(entry, "name", None) or getattr(entry, "parameter", None)
                        if name and name.lower() in ("readout_error", "assignment_error"):
                            val = getattr(entry, "value", None)
                            # readout_error usually a single number p for binary readout
                            if val is not None:
                                # build symmetric readout matrix
                                p = float(val)
                                mat = [[1 - p, p], [p, 1 - p]]
                                readouts_found[qi] = mat
            except Exception:
                pass

        # Add readout errors
        for q, mat in readouts_found.items():
            try:
                nm.add_readout_error(ReadoutError(mat), [q])
            except Exception:
                # older/newer APIs may have different signature
                try:
                    nm.add_readout_error(ReadoutError(mat), [q], warnings=warnings)
                except Exception:
                    pass

    return nm