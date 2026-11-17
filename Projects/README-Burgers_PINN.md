This  is  a  personal mini project on  developing  a hybrid PINN-based surrogate mdoel for the Inviscid Burgers' equation
The first part is a numerical pseudospectral solver whcih uses spectral numerics for the derivatives and a simple forward euler time integration
The  solution of the solver's history is used as the input training  data set for the PINN.
The autograd calculates the derivatives and the optimizer tries to minimize the  physics+ data losses (convergence  ~ 1e-3)
The final weights of the model are saved. and upon re-initialization of the saved model, the PINN is run again on the final input states
The  solutions are compared at the desired time duration.

Observations: PINN-surrogate does really well in the Linear regime but not the non-linear (shock-wave formation) as expected.
