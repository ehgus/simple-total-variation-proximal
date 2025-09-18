# Tatal variation Optimization toolkit

This package provides optimization tools for image processing and regularization, including Lp unit ball projections and total variation regularizers.

## Features

- **LpUnitBall**: Projections onto Lp unit balls for p = 0, 1, 2, and ∞
- **LpTotalVariation**: Total variation regularizers with different norms
- **OptProjection/OptRegularizer**: Base classes for optimization operators

## Running the Example

To test the TV denoising functionality:

```matlab
run('script_tv_denoising_test.m')
```

## References

- Benchettou, Oumaima, Abdeslem Hafid Bentbib, and Abderrahman Bouhamidi. "An accelerated tensorial double proximal gradient method for total variation regularization problem." Journal of Optimization Theory and Applications 198.1 (2023): 111-134.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM journal on imaging sciences.