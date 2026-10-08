# SLAI v2.3 STEM — Academic and Normative Grounding

The subsystem follows the boundary **STEM computes; Reasoning infers; Simulation evolves; Optimization searches**.

## Cross-cutting numerical reliability
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
- IEEE 754-2019. *IEEE Standard for Floating-Point Arithmetic*.
- Wilson, G., et al. (2014). Best Practices for Scientific Computing. *PLOS Biology*, 12(1), e1001745.
- Sandve, G. K., et al. (2013). Ten Simple Rules for Reproducible Computational Research. *PLOS Computational Biology*, 9(10), e1003285.

## `math/algebra.py`
- Geddes, K. O., Czapor, S. R., & Labahn, G. (1992). *Algorithms for Computer Algebra*. Springer.
- Bronstein, M. (2005). *Symbolic Integration I: Transcendental Functions* (2nd ed.). Springer.

## `math/calculus.py`
- Griewank, A., & Walther, A. (2008). *Evaluating Derivatives* (2nd ed.). SIAM.
- Bronstein (2005), above.

## `math/numerical_methods.py`
- Trefethen, L. N., & Bau, D. III. *Numerical Linear Algebra*. SIAM.
- Quarteroni, A., Sacco, R., & Saleri, F. (2007). *Numerical Mathematics*. Springer.
- Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving Ordinary Differential Equations I*. Springer.
- LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM.
- Davis, P. J., & Rabinowitz, P. (1984). *Methods of Numerical Integration* (2nd ed.). Academic Press.

## `math/statistics.py`
- Monahan, J. F. (2011). *Numerical Methods of Statistics* (2nd ed.). Cambridge University Press.
- Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). Algorithms for Computing the Sample Variance: Analysis and Recommendations. *The American Statistician*, 37(3), 242–247.

## `units/dimensions.py`, `units/unit_system.py`, `stem_types.py`
- BIPM. *The International System of Units (SI)*, 9th ed., current version.
- ISO 80000-1:2022. *Quantities and units — Part 1: General*.
- Kennedy, A. J. (1996). *Programming Languages and Dimensions*. University of Cambridge Computer Laboratory TR-391.
- Kennedy, A. (1997). Relational Parametricity and Units of Measure. POPL '97.
- Buckingham, E. (1914). On Physically Similar Systems. *Physical Review*, 4, 345.

## `uncertainty.py`
- JCGM 100:2008. *Guide to the Expression of Uncertainty in Measurement (GUM)*.
- JCGM GUM-6:2020. *Developing and Using Measurement Models*.
- JCGM 101:2008. *Propagation of Distributions Using a Monte Carlo Method*.

## Domain modules
- `biology.py`: Edelstein-Keshet, L. (2005), *Mathematical Models in Biology*; Segel & Edelstein-Keshet (2013), *A Primer on Mathematical Models in Biology*.
- `physics.py`: Landau, Páez & Bordeianu, *Computational Physics*; current CODATA fundamental constants via the inherited SLAI physics engine.
- `engineering.py`: Schäfer, M. (2022), *Computational Engineering — Introduction to Numerical Methods*; Quarteroni et al. (2007).
- `computer.py`: Cormen, Leiserson, Rivest & Stein (2022), *Introduction to Algorithms* (4th ed.).

## Template ownership
`templates/` is subsystem-owned data. `utils/temp_loader.py` is the sole canonical loader. Domain modules and the future `STEMAgent` may call this API; the Agent should never depend on template filesystem paths.
