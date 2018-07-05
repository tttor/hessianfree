# optimizers.md

# compute_update(): outline
(1) get step direction
  compute_gradient
  step direction via gauss-newton cg
  cg backtracking
  update damping param
(2) get step length
(3) update param

# gauss-newton cg
## @scipy
* https://docs.scipy.org/doc/scipy/reference/optimize.minimize-newtoncg.html
* this is not trivial,
  we have to pass fn to the function, in nets this fn is the whole neural nets
```
hessp : callable, optional
Hessian of objective function times an arbitrary vector p. Only for Newton-CG, trust-ncg, trust-krylov, trust-constr. Only one of hessp or hess needs to be given. If hess is provided, then hessp will be ignored. hessp must compute the Hessian times an arbitrary vector:

hessp(x, p, *args) ->  ndarray shape (n,)

where x is a (n,) ndarray, p is an arbitrary vector with dimension (n,) and args is a tuple with the fixed parameters.
```

## @pytorch
https://github.com/pytorch/pytorch/issues/1359
https://github.com/tfrerix/proxprop


## 1st-order cg
https://github.com/ikostrikov/pytorch-trpo/blob/master/conjugate_gradients.py

https://github.com/torch/optim/blob/master/doc/algos.md#optim.cg
https://github.com/torch/optim/blob/master/cg.lua

https://www.upwork.com/job/Implement-conjugate-gradient-optimization-algorithm-neural-network-framework_~01f058eabdf455ece6/
