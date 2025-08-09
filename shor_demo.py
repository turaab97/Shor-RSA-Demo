#!/usr/bin/env python3
"""
Shor's Algorithm (Educational Demo, 2025)
Author: Syed Ali Turab

What this file does
-------------------
- Factors a small RSA-like modulus N using the *mechanism* of Shor's algorithm:
  the quantum **order finding** subroutine + classical post-processing.
- Then shows a tiny end-to-end RSA "break": derive d and decrypt a sample message.

This is a simulator demo. It proves the concept, not real-world key recovery.
"""

import argparse
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import sympy
import cirq

# ----------------------------- Utilities ------------------------------------

def int_from_bits_lsb_first(bits) -> int:
    """Convert a measured bit array (LSB-first) to an integer."""
    bits = np.asarray(bits, dtype=int)
    return int(np.dot(bits, 1 << np.arange(bits.shape[-1])))

# ------------------------ Modular Exponentiation Gate ------------------------

class ModularExp(cirq.ArithmeticGate):
    """
    Arithmetic gate computing:
        |target>|exponent>  ->  | (target * a^exponent mod N) > |exponent>

    - We keep 'a' and 'N' fixed as part of the gate.
    - Qubits are grouped as two registers: target (size = target_bits), exponent (size = exponent_bits).
    - This gate is used controlled-by the exponent superposition during phase estimation.
    """

    def __init__(self, a: int, N: int, target_bits: int, exponent_bits: int):
        super().__init__(num_qubits=target_bits + exponent_bits)
        self.a = a
        self.N = N
        self.t_bits = target_bits
        self.e_bits = exponent_bits

    def registers(self):
        # lsb-first registers
        return cirq.Register("t", self.t_bits), cirq.Register("e", self.e_bits)

    def with_registers(self, *regs):
        (t, e) = regs
        return ModularExp(self.a, self.N, len(t), len(e))

    def apply(self, t, e):
        # Given integers t (target) and e (exponent), output new target only.
        return ((t * pow(self.a, e, self.N)) % self.N, e)

# ---------------------------- Order Finding ----------------------------------

@dataclass
class OrderFindingResult:
    a: int
    r: Optional[int]
    y: int     # raw measurement integer
    shots: int

def make_order_finding_circuit(a: int, N: int) -> Tuple[cirq.Circuit, list, list]:
    """
    Build the phase-estimation-style circuit to learn the order r of a mod N.

    Registers:
      - target: holds integers modulo N (size L = ceil(log2 N))
      - expo: exponent/phase register (size E ~ 2L+3)
    """
    L = N.bit_length()
    E = 2 * L + 3

    target = [cirq.LineQubit(i) for i in range(L)]
    expo   = [cirq.LineQubit(L + i) for i in range(E)]

    c = cirq.Circuit()
    # Prepare |1> in target (multiplicative identity in mod-N arithmetic)
    c.append(cirq.X(target[0]))
    # Create uniform superposition in exponent register
    c.append(cirq.H.on_each(*expo))

    # Controlled modular exponentiation (ArithmeticGate handles lifting)
    modexp = ModularExp(a=a, N=N, target_bits=L, exponent_bits=E)
    c.append(modexp.on(*target, *expo))

    # Inverse QFT on exponent register to extract phase
    c.append(cirq.inverse(cirq.qft(*expo, without_reverse=True)))

    # Measure exponent
    c.append(cirq.measure(*expo, key="e"))
    return c, target, expo

def continued_fraction_to_period(y: int, m: int, a: int, N: int) -> Optional[int]:
    """
    Given measurement y of m exponent qubits, estimate phase y/2^m and
    use continued fractions to recover a candidate period r.
    """
    # Rational approximation of y / 2^m
    frac = sympy.nsimplify(y / (2 ** m), rational=True, maxsteps=64)
    if getattr(frac, "q", 0) == 0:
        return None
    r = int(frac.q)
    # Sanity + small multiples if needed
    for k in range(1, 5 * N):
        cand = r * k
        if pow(a, cand, N) == 1:
            return cand
    return None

def quantum_order_finder(a: int, N: int, repetitions: int = 5, seed: Optional[int] = None) -> OrderFindingResult:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if math.gcd(a, N) != 1:
        # Trivial factor already exists; the "order" degenerates.
        return OrderFindingResult(a=a, r=1, y=0, shots=0)

    circuit, target, expo = make_order_finding_circuit(a, N)
    sim = cirq.Simulator()

    last_y = 0
    for shot in range(1, repetitions + 1):
        res = sim.run(circuit, repetitions=1)
        bits_lsb = res.measurements["e"][0][::-1]  # cirq returns msb-first; flip to lsb-first
        y = int_from_bits_lsb_first(bits_lsb)
        last_y = y
        m = len(expo)
        r = continued_fraction_to_period(y, m, a, N)
        if r is not None and r % 2 == 0 and pow(a, r // 2, N) != N - 1:
            return OrderFindingResult(a=a, r=r, y=y, shots=shot)

    return OrderFindingResult(a=a, r=None, y=last_y, shots=repetitions)

# ------------------------------- Shor Wrapper --------------------------------

def shor_factor_once(N: int, max_tries: int = 25, seed: Optional[int] = None) -> Optional[int]:
    """
    Try to recover a non-trivial factor of N using Shor's approach.
    Returns a single factor p if found, else None.
    """
    if N % 2 == 0:
        return 2
    if sympy.isprime(N):
        return None

    # Prime-power quick check
    for k in range(2, int(math.log2(N)) + 1):
        root = round(N ** (1 / k))
        if root ** k == N:
            return root

    rng = random.Random(seed)
    for _ in range(max_tries):
        a = rng.randrange(2, N - 1)
        g = math.gcd(a, N)
        if g > 1:
            return g  # lucky gcd shortcut

        of = quantum_order_finder(a, N, repetitions=6, seed=rng.randrange(10**9))
        if of.r is None or of.r % 2 == 1:
            continue
        y = pow(a, of.r // 2, N)
        p = math.gcd(y - 1, N)
        if 1 < p < N:
            return p
        p = math.gcd(y + 1, N)
        if 1 < p < N:
            return p
    return None

# ------------------------------- RSA Toy Demo --------------------------------

def rsa_small_demo(N: int, e: int = 65537, m: int = 8, seed: Optional[int] = 42) -> None:
    """
    End-to-end: factor N with Shor (simulated), compute private key d, and decrypt a sample message.
    """
    print(f"[Input] N={N}")
    p = shor_factor_once(N, seed=seed)
    if p is None:
        print(f"[Shor] Failed to find a factor of N={N}. Try another N (e.g., 15, 21, 33) or re-run.")
        return
    q = N // p
    print(f"[Shor] Found non-trivial factor p={p} of N={N} (q={q})")

    phi = (p - 1) * (q - 1)
    if math.gcd(e, phi) != 1:
        # choose a small e that is coprime to phi
        e = 17 if math.gcd(17, phi) == 1 else 3
    d = int(sympy.mod_inverse(e, phi))
    print(f"[RSA]  Using e={e}, computed d={d}  (mod phi(N)={phi})")

    if not (0 < m < N):
        m = min(8, N - 1)
    c = pow(m, e, N)
    m2 = pow(c, d, N)
    print(f"[RSA]  Message m={m} -> ciphertext c={c} -> decrypted m' = {m2}  ({'success' if m2 == m else 'mismatch'})")

# ---------------------------------- Main -------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Shor's Algorithm demo (simulated) + toy RSA break")
    parser.add_argument("--N", type=int, default=21, help="Composite modulus to factor (small: 15, 21, 33, ...)")
    parser.add_argument("--message", type=int, default=8, help="Toy RSA message m (0 < m < N)")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for reproducibility")
    args = parser.parse_args()
    rsa_small_demo(N=args.N, m=args.message, seed=args.seed)

if __name__ == "__main__":
    main()
