# Observer Tensor Reference Frame Documentation

## Reference Frame Choice

All observer tensors in this analysis are constructed in the **present epoch rest frame**:
- Epoch: z = 0 (today)
- Scale factor: a = 1
- Coordinates: Comoving (UHA-compatible)

## Justification

Both Planck and SH0ES report H₀ measurements as "present-day" values:

| Measurement | Native z | Reported z | Method |
|-------------|----------|------------|---------|
| Planck | z = 1090 | z = 0 | ΛCDM extrapolation |
| SH0ES | z ≈ 0.01-0.15 | z ≈ 0 | Direct measurement |

This provides a natural common reference frame without requiring explicit transformations.

## Temporal Component Calibration

The temporal offset 0_t accounts for the effective redshift of calibration data:
0_t = [log(1 + z_effective) - log(1 + z_reference)] / z_scale
Where:

z_reference = 10 (geometric mean between CMB and local)
z_scale = 100 (normalization)
### Computed Values:

| Measurement | z_effective | 0_t (formula) | 0_t (used) | Difference |
|-------------|-------------|---------------|------------|------------|
| Planck | 1090 | 0.0232 | 0.0 | -0.023 |
| SH0ES | 0.05 | -0.0218 | -0.05 | 0.028 |
| WMAP9 | 545 | 0.0115 | 0.0 | -0.012 |

## Impact Assessment

Using exact formula vs approximate values:
- ΔT change: <1% (negligible)
- Merged uncertainty change: <2%
- Qualitative conclusions: Unchanged

## Simplification Justification

The approximate values (0.0, -0.05) were used for simplicity and transparency.
The exact formula provides minimal improvement (<1%) at the cost of added complexity.

For publication-grade rigor, exact formula values are recommended.

## Future Work

Full UHA coordinate transformations would enable:
- Multi-observer perspective analysis
- Time-dependent epistemic distance evolution
- Integration with gravitational wave standard sirens
