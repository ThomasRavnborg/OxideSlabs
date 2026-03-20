# Perovskite DFT Workflow (GPAW / SIESTA)

This repository contains Python tools for performing **density functional theory (DFT)** calculations on perovskite structures using the **Atomic Simulation Environment (ASE)**. The code supports both **plane-wave (PW)** and **localized basis (LCAO)** approaches via two widely used DFT codes:

- GPAW (plane-wave / real-space grid)
- SIESTA (localized atomic orbitals)

The workflow is designed for efficient exploration of **structural instabilities in perovskite materials**, particularly in **thin films and slabs**. While originally developed for perovskites, the framework can in principle operate on **any ASE `Atoms` object**.

---

# Motivation

Perovskite materials often exhibit **structural instabilities**, such as soft phonon modes associated with octahedral rotations, polar distortions, or antiferrodistortive patterns. These instabilities strongly influence properties such as:

- ferroelectricity  
- dielectric response  
- structural phase transitions  

When studying **surfaces, interfaces, and thin films**, these instabilities may change significantly with:

- epitaxial **strain**
- **electric fields**
- **surface termination**
- **film thickness**

Standard plane-wave DFT calculations can become computationally expensive for large **slab geometries**.

This repository demonstrates a workflow where **localized orbital methods (LCAO)** can drastically **reduce computational cost** while maintaining **comparable qualitative accuracy** for phonon instabilities.

---

# Features

- Supports both **PW (GPAW)** and **LCAO (SIESTA)** calculations
- Built around **ASE**
- Works with **bulk and slab geometries**
- Modular Python workflow
- Automated workflow including:

1. **Structural relaxation**
2. **Electronic band structure**
3. **Phonon dispersion**
4. **Frozen phonon energy landscape mapping**

The workflow is particularly useful for studying **unstable phonon modes** and their evolution under different physical conditions.

---

# Workflow

The standard workflow proceeds as follows:

## 1. Structure Relaxation

The initial structure (typically a perovskite bulk or slab) is relaxed until forces converge.

Output:
- relaxed ASE atoms object (.xyz) and relaxation trajectory (.traj)
- density matrix (.DM), forces (.FA) and hamiltonians (.HSX)

---

## 2. Electronic Band Structure

The electronic band structure and density of states (DOS) is calculated along a high-symmetry path in reciprocal space.

Output:
- band structure data (.bands), density of states (.DOS) and projected density of states (.PDOS)

---

## 3. Phonon Dispersion

Using finite displacements (Phonopy), the phonon dispersion is calculated.

Output:
- identify **unstable modes (imaginary frequencies)**
- locate structural instabilities

---

## 4. Frozen Phonon Calculations

For unstable modes, the code performs **frozen phonon calculations**:

- atoms are displaced along the eigenvector of the unstable mode
- total energy is calculated as a function of displacement amplitude

This produces an **energy landscape** that reveals:

- double-well potentials
- structural phase transitions
- coupling between distortions

---

# GPAW vs SIESTA

This repository allows calculations using either GPAW or SIESTA, which differ in their treatment of electronic wavefunctions.

## GPAW (Plane Waves / Real-Space Grid)

- systematic basis set
- controlled by grid spacing / plane-wave cutoff
- generally **higher accuracy**
- computational cost scales strongly with system size

Advantages:

- reliable convergence
- widely used for benchmark calculations

Disadvantages:

- expensive for large slabs

---

## SIESTA (Localized Atomic Orbitals)

- basis set built from **localized orbitals**
- computational effort scales more favorably with system size
- well suited for **large systems and surfaces**

Advantages:

- **much faster calculations**
- lower memory requirements

Disadvantages:

- results depend on **basis set quality**
- less systematic convergence than plane-wave methods

---

# Main Idea

The central idea demonstrated here is that:

> **LCAO calculations can reproduce the qualitative physics of soft phonon modes while reducing computational cost dramatically compared to plane-wave calculations.**

For studies involving:

- many slab thicknesses
- multiple strain states
- different surface terminations
- frozen phonon energy landscapes

this speed improvement enables **large parameter sweeps that would otherwise be impractical**.

---

# Example Applications

The workflow can be used to study:

- perovskite **surface instabilities**
- **strain-dependent soft modes**
- **thickness dependence** in slabs
- energy landscapes of unstable phonons