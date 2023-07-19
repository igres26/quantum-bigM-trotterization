from qibo import gates
from qibo.models import Circuit
from qibo import hamiltonians
import numpy as np
import scipy as sp
from functions import *
import json


class Trotterize:
    """Class that handles the Trotterization of an adiabatic evolution given a problem Hamiltonian in the X basis.

    Args:
        nqubits (int): number of qubits of the system.
        hp (qibo.hamiltonians.Hamiltonian: problem Hamiltonian encoding the solution.
        order (int): order of the Trotterization, only 1 and 2 allowed.

    """
    def __init__(
        self,
        nqubits,
        hp,
        order=1,
    ):
        self.nqubits = nqubits
        self.hp = hp
        self.x = symbols(" ".join((f"x{i}" for i in range(0, self.nqubits))))
        self.dict_x = {str(xx):xx for xx in self.x}
        self.circuitX, self.circuitXX, self.circuitZ, self.paramsX, self.paramsXX, self.paramsZ = self.trotter_circuits()
        if order == 1:
            self.o = 1
        elif order == 2:
            self.o = 0
        
    def trotter_circuits(self):
        """Generate general circuits needed for each Trotter step.

        """
        nqubits = self.nqubits
        circuitX = Circuit(nqubits)
        paramsX = []
        circuitXX = Circuit(nqubits)
        paramsXX = []
        for arg in self.hp.args:
            if len(arg.args) == 0:
                pass
            elif len(arg.args) == 2:
                circuitX.add(gates.RX(arg.args[1].target_qubit, 0))
                paramsX.append(float(arg.args[0]))
            elif len(arg.args) == 3:
                circuitXX.add(gates.RXX(arg.args[1].target_qubit, 
                                       arg.args[2].target_qubit, 
                                       0))
                paramsXX.append(float(arg.args[0]))
            else:
                print('WARNING: The Hamiltonian has unexpected terms! Check the expression given!\n')

        circuitZ = Circuit(self.nqubits)
        circuitZ.add([gates.RZ(i, 0) for i in range(nqubits)])
        paramsZ = [1/(2*self.nqubits) for i in range(nqubits)]
        
        return circuitX, circuitXX, circuitZ, np.array(paramsX), np.array(paramsXX), np.array(paramsZ)
    
    def build_trotter_step(self, circuit, t, T, dt):
        """Set the parameters for the trotter step used to advance from time t, to time t+dt

        """
        circuit = Circuit(self.nqubits)
        self.circuitZ.set_parameters(-dt*(1-t/T)*(2**self.o)*self.paramsZ)
        self.circuitX.set_parameters(dt*(t/T)*(2**self.o)*self.paramsX)
        self.circuitXX.set_parameters(dt*(t/T)*(2**self.o)*self.paramsXX)
        circuit += self.circuitZ
        circuit += self.circuitX
        circuit += self.circuitXX
        if self.o == 0:
            circuit.add(self.circuitXX.queue[::-1])
            circuit.add(self.circuitX.queue[::-1])
            circuit.add(self.circuitZ.queue[::-1])
        return circuit.copy(deep=True)
        
    def build_trotter_circuit(self, T, dt):
        """Assemble the full Trotterized circuit by concatenating together all the needed trotter steps.

        """
        circuit = Circuit(self.nqubits)
        for t in np.arange(dt, T+dt, dt):
            circuit += self.build_trotter_step(circuit, t, T, dt)
        #circuit.add([gates.GPI2(i, -np.pi/2) for i in range(nqubits)])
        circuit.add([gates.H(i) for i in range(self.nqubits)])
        for i in range(self.nqubits):
            circuit.add(gates.M(i))
        return circuit


def add_rz(target, rotation):
    d = {}
    d["gate"] = "rz"
    d["target"] = target[0]
    d["rotation"] = rotation[0]
    return d


def add_rx(target, rotation):
    d = {}
    d["gate"] = "rx"
    d["target"] = target[0]
    d["rotation"] = rotation[0]
    return d


def add_rxx(targets, rotation):
    d = {}
    d["gate"] = "xx"
    d["targets"] = [*targets[::-1]]
    d["rotation"] = rotation[0]
    return d


def add_h(target):
    d = {}
    d["gate"] = "h"
    d["target"] = target[0]
    return d


def add_gpi2(target, phase):
    d = {}
    d["gate"] = "gpi2"
    d["phase"] = phase
    d["target"] = target[0]
    return d


def add_gpi(target, phase):
    d = {}
    d["gate"] = "gpi2"
    d["phase"] = phase
    d["target"] = target[0]
    return d


def add_ms(targets, phases, angle):
    d = {}
    d["gate"] = "ms"
    d["targets"] = [*targets[::-1]]
    d["phases"] = [*phases[::-1]]
    d["angle"] = angle
    return d


def IonQ_parser(c, shots, target, name, f="ionq.circuit.v0", gateset="qis", noise=None):
    """Parser that generates a Qibo circuit and returns a json file with experimental
    details to be executed on an IonQ quantum device.
    Args:
        c (qibo.models.Circuit): circuit to be executed on IonQ hardware.
        shots (int): number of shots to request.
        target (str): name of the target device.
        f (str): format for the parser.
        gateset (str): whether to use the 'qis' or 'native' gateset.
        noise (str): noise model to use in case of simulation in IonQ servers.
    
    Returns:
        json file with the instructions needed to execute the circuit.
    
    """
    json_dict = {}
    json_dict["lang"] = "json"
    json_dict["shots"] = shots
    json_dict["target"] = target
    if noise:
        json_dict["noise"] = {"model": noise, "seed": 100}
    json_dict["name"] = name
    body = {}
    body["format"] = f
    body["gateset"] = gateset
    body["qubits"] = c.nqubits
    q_phases = np.zeros(c.nqubits)
    circuit = []
    twoq = 0

    if gateset == "qis":
        for gate in c.queue:
            if gate.name == 'rz':
                circuit.append(add_rz(gate.target_qubits, gate.parameters))
            elif gate.name == 'rx':
                circuit.append(add_rx(gate.target_qubits, gate.parameters))
            elif gate.name == 'rxx':
                circuit.append(add_rxx(gate.target_qubits, gate.parameters))
            elif gate.name == 'h':
                circuit.append(add_h(gate.target_qubits))
            else:
                print("Gate skipped! Only H, RZ, RX and RXX supported at this stage!")
    elif gateset == "native":
        for gate in c.queue:
            #print(q_phases)
            if gate.name == 'rz':
                q_phases[gate.target_qubits[0]] -= gate.parameters[0]
            elif gate.name == 'rx':
                phase = q_phases[gate.target_qubits[0]]+(0.75*2*np.pi)
                circuit.append(add_gpi2(gate.target_qubits, (phase/(2*np.pi))%1))
                q_phases[gate.target_qubits[0]] -= -gate.parameters[0]
                phase = q_phases[gate.target_qubits[0]]+(0.25*2*np.pi)
                circuit.append(add_gpi2(gate.target_qubits, (phase/(2*np.pi))%1))
            elif gate.name == 'rxx':
                phases = [(q_phases[gate.target_qubits[0]]/(2*np.pi))%1,
                          (q_phases[gate.target_qubits[1]]/(2*np.pi))%1]
                angle = gate.parameters[0]
                for _ in range(int(((angle/(2*np.pi))%1)//0.25)):
                    twoq += 1
                    circuit.append(add_ms(gate.target_qubits, phases, 0.25))
                twoq += 1
                circuit.append(add_ms(gate.target_qubits, phases, ((angle/(2*np.pi))%1)%0.25))

            elif gate.name == 'h':
                phase = q_phases[gate.target_qubits[0]]+np.pi/2
                circuit.append(add_gpi2(gate.target_qubits, (phase/(2*np.pi))%1))
            else:
                print("Gate skipped! Only H, RZ, RX and RXX supported at this stage!")
    else:
        raise ValueError('Not a valid Gateset. Only qis or native supported.')

    print(f'Number of native 2 qubit gates: {twoq}\n')
    body["circuit"] = circuit
    json_dict["body"] = body
    with open(f"{name}_{gateset}.json", "w") as fp:
        json.dump(json_dict, fp)


def success_probability(circuit, err1, err2, errM):
    """Estimate the probability that a given circuit execute flawlesly on device.
    Args:
        circuit (qibo.models.Circuit): circuit to be tested.
        err1 (float): fidelity of single qubit gates.
        err2 (float): fidelity of two qubit gates.
        errM (float): fidelity of readout.
    
    Returns:
        probability that no errors occur during circuit execution.
    
    """
    q1 = 0
    q2 = 0
    for gate in circuit.queue:
        if gate.name == 'rx':
            q1 += 2
        if gate.name == 'x':
            q1 += 1
        if gate.name == 'cz':
            q2 += 1
    return (err1**q1)*(err2**q2)*(errM**circuit.nqubits)
    