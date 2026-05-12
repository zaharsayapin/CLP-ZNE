# Cyclic Layout Permutation based Zero Noise Extrapolation

Increasing the utility of currently available Noisy Intermediate-Scale Quantum (NISQ) devices requires developing efficient methods to mitigate hardware errors. In this work we propose a novel Cyclic Layout Permutations based Zero Noise Extrapolation (CLP-ZNE) protocol for such a task. The method leverages the inherent non-uniformity of gate errors in NISQ hardware to extrapolate the expectation value, averaged over cyclic circuit layout permutations, to the level of zero noise. In contrast to the previous layout permutation based approaches, for n qubit circuit CLP-ZNE requires execution of only $O(n)$ and at most $O(n^2)$ different circuit layouts for circuits of one-dimensional and arbitrary connectivity, respectively. When benchmarked against noise channels modeling the IBM Torino quantum computer, the method reduces a typical error in expectation values of n=12 qubit circuits by an order of magnitude, outperforming standard unitary folding ZNE. By demonstrating the ability to mitigate noise of real hardware specifications, including both depolarizing and $T_1/T_2$ relaxation processes, these results give evidence for the applicability of CLP-ZNE to present-day NISQ processors.

The repository contains the numerical simulations for the paper https://arxiv.org/abs/2511.02901.

## Citation
If you use the numerical results in this repository, please cite the following paper:

```bibtex
@article{sayapin2025zero,
  title={Zero-Noise Extrapolation via Cyclic Permutations of Quantum Circuit Layouts},
  author={Sayapin, Zahar and Rabinovich, Daniil and Korolev, Nikita and Lakhmanskiy, Kirill},
  journal={arXiv preprint arXiv:2511.02901},
  year={2025}
}
