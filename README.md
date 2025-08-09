# Shor vs RSA (2025) — Minimal, Runnable Demo

**Author:** Syed Ali Turab
**Original talk (2019):** https://www.youtube.com/watch?v=guGM1l7KneM

This repo is a 2025 refresh of my 2019 presentation on *how Shor's algorithm breaks RSA* — not by “hacking” keys directly, but by factoring the RSA modulus \(N = p \cdot q\) in **polynomial time** on a *fault-tolerant* quantum computer. We obviously don't have that hardware yet, so this is a small, educational **simulator** demo that shows the exact moving parts:

- the **quantum order-finding** core (via phase estimation),
- the **classical post-processing** (GCD trick to pop factors),
- and then a **toy RSA key break** once a non-trivial factor is found.

> TL;DR: When scalable, error-corrected quantum machines arrive, Shor's algorithm kills RSA and ECC. This repo shows *why*, end-to-end, on tiny integers you can run today.

---

## What I did in 2019 vs. what changed by 2025

**2019 (my presentation):**
- Explained why RSA’s security reduces to factoring \(N\).
- Walked through Shor’s order-finding idea and used a small-number demo to make it tangible.
- Framed the threat model: once we get large, error-corrected quantum computers, factoring becomes efficient.

**2025 (this repo):**
- Clean, dependency-light **Cirq** implementation of the order-finding subroutine (the hard part).
- Robust **classical pre/post-processing**: GCD shortcuts, continued fractions, and the \(a^{r/2} \pm 1\) trick.
- A **toy RSA flow**: generate a tiny RSA modulus, “quantum-factor” it on the simulator, then derive the private key and decrypt.
- Everything is **well-commented** so you can actually teach from it or extend it.

> Reality check (2025): You **cannot** break RSA-2048 on a classical laptop. This repo is about pedagogy: it shows the *mechanism* that will break RSA at scale.

---

## Quickstart

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the demo (defaults to N=21)
python shor_demo.py

# Try a different modulus
python shor_demo.py --N 33   # 3*11
python shor_demo.py --N 15   # 3*5
```

Expected output looks like:
```
[Shor] Found non-trivial factor p=3 of N=21 (q=7)
[RSA]  Using e=65537, computed d=...  (mod phi(N))
[RSA]  Message m=8 -> ciphertext c=... -> decrypted m' = 8  (success)
```

---

## How it works

1. **Pick a random base `a` with gcd(a, N) = 1**.  
2. **Quantum subroutine:** Use phase estimation to learn the **order** `r` such that \(a^r \equiv 1 \ (mod\ N)\).  
3. **Classical glue:** If `r` is even and \(a^{r/2} \not\equiv -1 \ (mod\ N)\), then compute  
   \( \gcd(a^{r/2} - 1, N) \) or \( \gcd(a^{r/2} + 1, N) \) to get a non-trivial factor of `N`.  
4. **Break RSA:** Once you have \(p\) and \(q\), compute \(\varphi(N)\), invert `e` to get `d`, and decrypt.

The **quantum win** is Step 2: classically, order finding is generally hard; quantumly it becomes efficient via the Quantum Fourier Transform.

---

## Repo layout

```
.
├── LICENSE
├── README.md
├── requirements.txt
├── shor_demo.py       # runnable, heavily commented
└── .gitignore
```

---

## Teaching notes

- Keep `N` small (15, 21, 33) so the simulator finishes quickly.
- The measurement-to-period step uses continued fractions; noisy samples may require a few repetitions.
- For a classroom, set a fixed random seed, or pre-pick `a` that gives an even period.

---

## License

MIT — see `LICENSE`.
