import numpy as np
from qiskit.quantum_info import SparsePauliOp

def sherrington_kirkpatrick_model(n, h=1.0, seed=None):
    """
    Generates Sherrington-Kirkpatrick model with transverse field.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate coupling matrix
    J_matrix = np.random.normal(0, 1, (n, n))
    np.fill_diagonal(J_matrix, 0)
    J_matrix = (J_matrix + J_matrix.T)
    
    # Pre-allocate arrays for efficiency
    num_zz_terms = n * (n - 1) // 2
    num_x_terms = n
    total_terms = num_zz_terms + num_x_terms
    
    pauli_labels = np.empty(total_terms, dtype=object)
    coefficients = np.zeros(total_terms)
    
    # Build ZZ terms
    term_idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if J_matrix[i, j] != 0:
                # Create Pauli string efficiently
                pauli_str = ['I'] * n
                pauli_str[i] = 'Z'
                pauli_str[j] = 'Z'
                pauli_labels[term_idx] = ''.join(pauli_str)
                coefficients[term_idx] = J_matrix[i, j] / np.sqrt(n)
                term_idx += 1
    
    # Build X terms
    for i in range(n):
        pauli_str = ['I'] * n
        pauli_str[i] = 'X'
        pauli_labels[term_idx] = ''.join(pauli_str)
        coefficients[term_idx] = h
        term_idx += 1
    
    return SparsePauliOp(pauli_labels, coefficients)