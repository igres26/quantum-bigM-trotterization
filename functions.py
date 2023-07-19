import numpy as np
import os
from qibo import hamiltonians
from qibo.symbols import X, Z
from qibo.models.evolution import AdiabaticEvolution
from qibo.models.variational import QAOA
from sympy import symbols
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.simplify.simplify import simplify
import matplotlib.pyplot as plt


def get_filenames(folder, typeM, nqubits):
    """Function that collects all the files in a folder into a list for further processing.
    Args:
        folder (str): name of the folder where the instances are saved.
        typeM (str): type of big M computation method. Can be 'solution' to get the solution files.
        nqubits (int): number of qubits to study.
    
    Returns:
        files (list): list with the paths to all the files to study.
    
    """
    files = []
    directory = folder+'/'+typeM+'/'+str(nqubits)
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        files.append(f)
    return files


def get_sym_obj(file, dict_x):
    """Get the symbolic representatio of the instance of a given file.
    Args:
        file (str): path to the file to study.
        dict_x (dict): dictionary needed to parse the expression into symbols.

    Returns:
        obj: sympy expression of the QUBO instance.

    """
    f = open(file, "r")
    obj = []
    for line in f.readlines()[5:]:
        if line == 'Subject To\n':
            break
        obj.append(line[6:-1])
    obj = ' '.join(obj)
    obj = obj.replace('^', '**')
    obj = obj.replace('[', '(')
    obj = obj.replace(']', ')')
    transformations = (standard_transformations + (implicit_multiplication_application,))
    obj = parse_expr(obj, dict_x, transformations=transformations)
    return obj

def get_cons_obj(file, dict_x):
    """Get the symbolic representatio of the instance of a given file.
    Args:
        file (str): path to the file to study.
        dict_x (dict): dictionary needed to parse the expression into symbols.

    Returns:
        obj: sympy expression of the QUBO instance.

    """
    f = open(file, "r")
    obj = []
    flag = 0
    for line in f.readlines()[5:]:
        if flag == 1:
            obj.append(line[6:-1])
            break
        if line == 'Subject To\n':
            flag = 1
            
    obj = ' '.join(obj)
    obj = obj.replace('^', '**')
    obj = obj.replace('[', '(')
    obj = obj.replace(']', ')')
    transformations = (standard_transformations + (implicit_multiplication_application,))
    obj = parse_expr(obj, dict_x, transformations=transformations)
    return obj


def substitutions(x, basis='z'):
    """Substitutions of the symbols in the instance into pauli matrices for quantum simulation.
    Args:
        x (list): list of the symbols used in the expression.

    Returns:
        s (list): list of the substitutions needed to go to pauli matrices.

    """
    s = []
    for i in range(len(x)):
        s.append((x[i]*x[i], x[i]))
    if basis == 'z':
        for i in range(len(x)):
            s.append((x[i], (1-Z(i))/2))
    elif basis == 'x':
        for i in range(len(x)):
            s.append((x[i], (1-X(i))/2))
    return s


def get_maxmin_eigenvalues(expression):
    """Compute the minimum and maximum eigenvalues for a given expression for normalization.
    Args:
        expression: sympy expression to compute the minimum and maximum eigenvalues.

    Returns:
        min, max (float): minimum and maximum eigenvalues of the expression.

    """
    m = hamiltonians.SymbolicHamiltonian(expression)
    m_eigs = m.eigenvalues()
    mx = max(m_eigs)
    mn = min(m_eigs)
    return mx, mn


def get_max_feasible(expression, constraints):
    """Compute the minimum and maximum eigenvalues for a given expression for normalization.
    Args:
        expression: sympy expression to compute the minimum and maximum eigenvalues.

    Returns:
        min, max (float): minimum and maximum eigenvalues of the expression.

    """
    m = hamiltonians.SymbolicHamiltonian(expression)
    m_eigs = m.eigenvalues()
    mx = max(m_eigs)
    mn = min(m_eigs)
    return mx, mn


def get_sym_ham(sym_obj, x, basis='z'):
    """Get symbolic hamiltonian from sympy expression, normalized.
    Args:
        sym_obj: symbolic expression of the QUBO instance.
        x (list): list of symbols that appear in the expression

    Returns:
        ham (SymbolicHamiltonian): hamiltonian qibo object that encodes the qubo instance.

    """
    subs = substitutions(x, basis=basis)
    ham = simplify(sym_obj.subs(subs))
    max_eig, min_eig = get_maxmin_eigenvalues(ham)
    ham = hamiltonians.SymbolicHamiltonian(simplify((ham-min_eig)/(max_eig-min_eig)))
    return ham


def get_h0(nqubits, basis='z'):
    """Generate the initial hamiltonian for the adiabatic evolution/qaoa, normalized.
    Args:
        nqubits (int): number of qubits of the instance.

    Returns:
        ham0 (SymbolicHamiltonian): X hamiltonian used for initial hamiltonian or mixer hamiltonian.

    """
    if basis == 'z':
        ham0 = sum((0.5 * (1 - X(i))) for i in range(nqubits))
    elif basis == 'x':
        ham0 = sum((0.5 * (1 - Z(i))) for i in range(nqubits))
    ham0 = hamiltonians.SymbolicHamiltonian(simplify(ham0/nqubits))
    return ham0


def get_solution(file, nqubits):
    """Get the solution of the instance from the solution files.
    Args:
        file (str): path to the solution file to analyze.
        nqubits (int): number of qubits of the instance to go from binary to int.

    Returns:
        sol (int): number of the solution as if it was in binary.
        sol_bin (str): binary configuration of the solution.

    """
    f = open(file, "r")
    sol_bin = f.readline()
    sol_bin = sol_bin[1:-1:2]
    sol = 0
    for i in range(len(sol_bin)):
        sol += int(sol_bin[i])*2**(nqubits-1-i)

    return sol, sol_bin


def save_data(data, name_data):
    """General function to save some data.
    Args:
        data: data to be saved.
        name_data (str): name for the file where the data is to be saved.

    """
    np.save(f'data/{name_data}.npy', data)


def read_constraint(file, dict_x):
    """Get the symbolic representation of the constraints of a given file.
    Args:
        file (str): path to the file to study.
        dict_x (dict): dictionary needed to parse the expression into symbols.

    Returns:
        constraint: symbolic expression of the constraint.

    """
    f = open(file, 'r')
    f.readline()
    flag = 0
    obj = []
    constraint = []
    for line in f.readlines()[5:]:
        if line == 'Bounds\n':
            break
        if flag == 1:
            obj.append(line[5:-1])
        if line == 'Subject To\n':
            flag = 1

    for i in range(len(obj[:-1])):
        #obj[i] = ' '.join(obj[i])
        obj[i] = obj[i].replace('=', '-')
        transformations = (standard_transformations + (implicit_multiplication_application,))
        constraint.append(parse_expr(obj[i], dict_x, transformations=transformations))
    return constraint


def get_subspace(nqubits, x, constraint):
    """Get the subspace of possible solutions that satisfy the constraints.
    Args:
        nqubits (int): number of qubits of the instance.
        x (list): list of symbols that appear in the expression.
        constraint: sympy expression of the constraint.

    Returns:
        subspace: list of solutions that satisfy the constraints.

    """
    subspace = []
    for i in range(2**nqubits):
        cs = 0
        bini = format(i,f"0{nqubits}b")#[::-1]
        for cc in constraint:
            cons = cc
            for j in range(nqubits):
                cons = cons.subs(x[j], int(bini[j]))
            cs += cons
        #print(cons)
        if cs == 0:
            subspace.append(i)
    return subspace


def get_max_energy(subspace, ham, nqubits, x):
    """Get the minimum and maximum values of the objective that sastisfy the contraint.
    Args:
        subspace (list): list of solutions that satisfy the constraints.
        ham: symbolic expression of the objective function.
        constraint: symbolic expression of the constraint.

    Returns:
        m (float): maximum value of allowed instances
        s_max (int): solution that has the maximum value
        M (float): minimum value of allowed instances
        s_min (int): solution that has the minimum value

    """
    m = -10000000
    M = 100000000
    for s in subspace:
        h = ham
        bins = format(s,f"0{nqubits}b")[::-1]
        for j in range(nqubits):
            h = h.subs(x[j], int(bins[j]))
        #print(h, s)
        if h > m:
            m = h
            s_max = s
        if h < M:
            M = h
            s_min = s
    return m, s_max, M, s_min


def get_ar(probs, m, M, ham, nqubits, subspace, minimum=False, post=False):
    """Get the approximation ratios of a given probability vector of solutions.
    Args:
        probs (list): Probabilities of measuring each computational basis state.
        m (float): maximum value of allowed instances
        M (float): minimum value of allowed instances
        ham: symbolic expression of the objective function.
        nqubits (int): number of qubits of the instance.
        subspace (list): list of solutions that satisfy the constraints.
        minimum (bool): if to average or take the best approximation ratio of the measurements.
        post (bool): if to only average around the states that satisfy the constraint.

    Returns:
        ar (float): Value for the approximation ratio of a given vector of probabilities.

    """
    x = symbols(" ".join((f"x{i}" for i in range(0, nqubits))))

    if post:
        ar = 0
        p = 0
        for s in subspace:
            h = ham
            bins = format(s,f"0{nqubits}b")[::-1]
            for j in range(nqubits):
                h = h.subs(x[j], int(bins[j]))
            ar += probs[s]*abs((h-M)/(m-M))
            p += probs[s]
        return ar/p
    else:
        if minimum:
            ar = 0
            for i in range(len(probs)):
                if i not in subspace:
                    ar_t = 0
                else:
                    h = ham
                    bins = format(i,f"0{nqubits}b")[::-1]
                    for j in range(nqubits):
                        h = h.subs(x[j], int(bins[j]))
                    ar_t = abs((h-M)/(m-M))
                if ar < ar_t:
                    ar = ar_t
            return ar
        else:
            ar = 0
            for i in range(len(probs)):
                if i not in subspace:
                    ar += 0
                else:
                    h = ham
                    bins = format(i,f"0{nqubits}b")[::-1]
                    for j in range(nqubits):
                        h = h.subs(x[j], int(bins[j]))
                    ar += probs[i]*abs((h-M)/(m-M))
            return ar
