# Optimizers

# compute_update(): outline
(at hessianfree/hessianfree/optimizers.py)
* (1) get step direction
  * compute_gradient
  * step direction via gauss-newton cg
  * cg backtracking
  * update damping param
* (2) get step length (line search for learning rate)
* (3) update param
