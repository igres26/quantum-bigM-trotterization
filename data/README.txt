The folder contains QUBO formulation using different Ms (our method, qiskit method, babbush method), alongside with constrained 
formulations and solutions (both x and f(x)), for "easy instances", meaning they have a big relative gap, and 6 qubits.
25 instances for each dataset are present.

First layer directories:

NN_linear_deg5              --> unstructured dataset (most promising one in terms of adiabatic evolution simulation, as our gap ~ 0.1)
SPP_p15                     --> Set Partitioning Problem dataset (our gap ~ 0.01)
PO_sp500_part3_ra10_mult2   --> Portfolio Optimization dataset (our gap ~ 0.01)

Second layer directories:

constrained --> original constrained quadratic problem
solution --> solution point (x) and value (f(x)) of the constrained problem
ourM --> QUBO reformualtion using our method
qiskM --> QUBO reformualtion using qiskit method
babbushM --> QUBO reformualtion using babbush-inspired method
