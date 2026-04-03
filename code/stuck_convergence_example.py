#!/usr/bin/env python3
"""
Example of ORIGINAL BROKEN CODE - Demonstrates stuck convergence
DO NOT USE THIS - It's here to show what DOESN'T work!
"""

import numpy as np
import pandas as pd

def extract_observer_tensor_BROKEN(chain, probe_type):
    """
    ORIGINAL BROKEN TENSOR EXTRACTION
    This is what causes Δ_T to be stuck at 0.6255
    """
    # Only uses statistical moments - NO SPATIAL INFORMATION!
    h0_mean = chain['H0'].mean()
    h0_std = chain['H0'].std()
    omega_m_mean = chain['omega_m'].mean() if 'omega_m' in chain else 0.3
    omega_m_std = chain['omega_m'].std() if 'omega_m' in chain else 0.01
    n_eff = len(chain)
    
    # EMPIRICAL FORMULAS WITH NO THEORETICAL BASIS
    P_m = 0.95 * (1 - np.exp(-n_eff / 5000))
    O_t = np.tanh(0.1 * (1 + 0.1 * np.tanh((67.4 - h0_mean) / 2)))
    O_m = 0.3 * (1 - np.exp(-omega_m_std / 0.01))
    O_a = 0.5  # Just a constant!
    
    # This tensor NEVER CHANGES between iterations!
    return np.array([P_m, O_t, O_m, O_a])

def broken_refinement(planck_chain, shoes_chain):
    """
    BROKEN ITERATIVE REFINEMENT - Gets stuck!
    """
    print("DEMONSTRATING BROKEN CONVERGENCE:")
    print("=" * 50)
    
    alpha = 0.15  # Learning rate
    
    # Extract tensors (THESE NEVER CHANGE!)
    tensor_planck = extract_observer_tensor_BROKEN(planck_chain, 'planck')
    tensor_shoes = extract_observer_tensor_BROKEN(shoes_chain, 'shoes')
    
    print(f"Initial tensors:")
    print(f"  Planck: {tensor_planck}")
    print(f"  SH0ES:  {tensor_shoes}")
    
    # Try to refine (BUT IT DOESN'T WORK!)
    for iteration in range(6):
        # This is supposed to refine the tensor
        tensor_fresh = extract_observer_tensor_BROKEN(planck_chain, 'planck')
        
        # Learning update
        tensor_refined = tensor_planck + alpha * (tensor_fresh - tensor_planck)
        
        # Compute epistemic distance
        delta_t = np.linalg.norm(tensor_refined - tensor_shoes)
        
        print(f"Iteration {iteration+1}: Δ_T = {delta_t:.4f}  <-- STUCK!")
        
        # Update for next iteration (but nothing changes!)
        tensor_planck = tensor_refined
    
    print("\n❌ NO CONVERGENCE - Δ_T stuck at same value!")
    print("This is because tensors don't change without resolution parameter")
    
    return delta_t

def main():
    """
    Demonstrate the original problem
    """
    print("\n" + "="*60)
    print("ORIGINAL BROKEN METHOD - STUCK CONVERGENCE DEMO")
    print("="*60)
    
    # Generate test data
    np.random.seed(42)
    
    planck_chain = pd.DataFrame({
        'H0': np.random.normal(67.4, 0.5, 1000),
        'omega_m': np.random.normal(0.315, 0.007, 1000)
    })
    
    shoes_chain = pd.DataFrame({
        'H0': np.random.normal(73.0, 1.0, 1000),
        'omega_m': np.random.normal(0.3, 0.02, 1000)
    })
    
    # Run broken refinement
    final_delta = broken_refinement(planck_chain, shoes_chain)
    
    print("\n" + "="*60)
    print("WHAT'S WRONG:")
    print("="*60)
    print("2. No resolution parameter")
    print("3. Tensors based only on statistical moments")
    print("4. Same tensor extracted every iteration")
    print("5. Result: Δ_T stuck at 0.6255 forever!")
    
    print("\n" + "="*60)
    print("THE FIX:")
    print("="*60)
    print("This allows tensors to change and convergence to occur")
    print("See multiresolution_uha_tensor_calibration.py for working version")

if __name__ == "__main__":
    main()
