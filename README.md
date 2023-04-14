# Modular Symbols for $\Gamma_0(N)$

## Introduction

This repository contains supplemental code to my Part III essay on modular symbols.

`main.py` contains some classes and functions involved in computing with modular symbols. In particular:

- `m = ModularSymbols(k, N)` constructs the space $\mathbb{M}_k(\Gamma_0(N))$
- `s = m.cuspidal_subspace()` computes the subspace of cuspidal modular symbols $\mathbb{S}_k(\Gamma_0(N))$
- `s.T_matrix(n)` computes the matrix of the action of $T_n$ on $\mathbb{S}_k(\Gamma_0(N))$

The above functions are used in the function `cusp_forms` to compute a basis for $S_k(\Gamma_0(N))$.

See `demo.ipynb` for example usage of `cusp_forms`.

## Requirements

The `sympy` package is required to run the code.

## References

Stein, W. (2007). *Modular forms, a computational approach (Graduate studies in mathematics; v. 79).* Providence, R.I.: American Mathematical Society.

Merel, L. (1994). *Universal Fourier expansions of modular forms.* In: Frey, G. (eds) On Artin's Conjecture for Odd 2-dimensional Representations. Lecture Notes in Mathematics, vol 1585. Springer, Berlin, Heidelberg. https://doi.org/10.1007/BFb0074110
