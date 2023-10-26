from typing import Any
import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import Matrix, Symbol, lambdify, matrix_multiply_elementwise
import math


batch_matrix_vec = jax.vmap(lambda M, v: M @ v)
"""
[B, N, K] @ [B, K]
"""

class DRIFT:
    
    def __init__(self, variable, drift, initial_value=0., module = 'jax', integral_form=False, diagonal=True):
        
        self.diagonal_form = diagonal

        if integral_form:
            # if drift is given integral form
            self.drift_int = drift
            self.drift = sympy.diff(self.drift_int, variable)
        else:
            self.drift = drift
            self.drift_int = sympy.integrate(drift,variable)
            
        if not diagonal:
            assert self.drift @ self.drift_int == self.drift_int @ self.drift, "The drift must commute with it's integral"
            # This check is not neccesary for diagonal matrices, as they will always commute.
        
            self.solution_matrix = lambdify(variable, (self.drift_int-self.drift_int.limit(variable, initial_value)).exp(), module)
        else:
            # We can split up the exponential
            #self.solution_matrix = lambdify(variable, matrix_multiply_elementwise(self.drift_int.applyfunc(sympy.exp).transpose(),(-self.drift_int).applyfunc(sympy.exp).limit(variable,initial_value).transpose()), module)
            self.solution_matrix = lambdify(variable, (self.drift_int-self.drift_int.limit(variable,initial_value)).applyfunc(sympy.exp), module)

        self.drift_call = lambdify(variable, self.drift, module)

        if module == 'jax':
            self.drift_call = jax.vmap(self.drift_call)
            self.solution_matrix = jax.vmap(self.solution_matrix)
        
    def __repr__(self):
        return str(self.drift)

    def __call__(self, time, data) -> Any:
        if self.diagonal_form:
            return self.drift_call(time) * data
        else:
            return self.drift_call @ data

class DIFFUSION:
    
    def __init__(self, variable, diffusion, diffusion_matrix=None, initial_value=0., module='jax', integral_form=False, integral_decomposition='cholesky', diagonal=True, diagonal_diffusion=True):
    
        self.diagonal_form = diagonal and diagonal_diffusion

        if integral_form:
            
            if diffusion_matrix is not None:
                print("Diffusion matricies supplied with integral forms are ignored")
            
            self.diffusion_int = diffusion
            diffusion_term = sympy.diff(self.diffusion_int, variable)
            if integral_decomposition == 'cholesky':
                
                if not diagonal:
                    self.diffusion = diffusion_term.cholesky()
                    self.diffusion_matrix = Matrix.eye(self.diffusion.shape[0])
                else:
                    self.diffusion = diffusion_term.applyfunc(sympy.sqrt)
                    # TODO: ADD CHECK TO SEE IF THE SOLUTION IS VALID
                    self.diffusion_matrix = sympy.ones(*self.diffusion.shape)
                    
            elif integral_decomposition == 'ldl':
                if not diagonal:
                    self.diffusion, self.diffusion_matrix = diffusion_term.LDLdecomposition()
                    assert self.diffusion_matrix.diff(variable).is_zero_matrix, f"The diffusion matrix must not be dependent on {variable}"
                else:
                    print("TODO: LDL composition is not implemented for the diagonal form\n Performing cholesky decomposition instead")
                    
                    # TODO: ADD CHECK TO SEE IF THE SOLUTION IS VALID
                    self.diffusion = diffusion_term.applyfunction(sympy.sqrt)
                    
                    
                    self.diffusion_matrix = sympy.ones(*self.diffusion.shape)            
                    
            else:
                raise NotImplemented(f'Solution not implemented for the decomposition form {integral_decomposition}')
            
        else:
            
            self.diffusion = diffusion
            
            assert diffusion_matrix is not None, "A diffusion matrix must be given, consider using the identity matrix"
            
            self.diffusion_matrix = diffusion_matrix
            
            # Compute the integral based on the form of the diffusion and diffusion matrix
            # For matricies with a large dimensionality, it is recommended to use diagonal matricies
            # and pass them as a vector with diagonal=True, diagonal_diffusion=True
            def matrix_diagonal_product(matrix, diagonal):
                return Matrix(list(map(lambda x: x[0] * x[1], zip([matrix[:,i] for i in range(matrix.shape[0])],diagonal)))).reshape(*matrix.shape).transpose()
                
            if not diagonal and not diagonal_diffusion:
                self.diffusion_int = sympy.integrate(self.diffusion@(self.diffusion_matrix@self.diffusion.transpose()), variable)
            
            elif diagonal and diagonal_diffusion:
                self.diffusion_int = sympy.integrate(matrix_multiply_elementwise(self.diffusion, matrix_multiply_elementwise(self.diffusion_matrix, self.diffusion)), variable)
            
            
                
            elif not diagonal and diagonal_diffusion:
                
                self.diffusion_int = sympy.integrate(matrix_diagonal_product(self.diffusion, self.diffusion_matrix) @ self.diffusion.transpose(), variable)
                
                
            elif diagonal and not diagonal_diffusion:
                
                self.diffusion_int = sympy.integrate(matrix_diagonal_product(matrix_diagonal_product(self.diffusion_matrix, self.diffusion).transpose(), self.diffusion).transpose(), variable)
                
        self.solution_matrix = (self.diffusion_int-self.diffusion_int.limit(variable, initial_value))
        
        if not self.diagonal_form:
            
            # TODO: ADD CHECK TO SEE IF THE SOLUTION IS VALID
            self.decomposition = self.solution_matrix.cholesky()
            self.inv_decomposition = self.decomposition.inv()
            self.inv_covariance = self.solution_matrix.inv()
        else:
            
            self.decomposition = self.solution_matrix.applyfunc(sympy.sqrt)
            self.inv_decomposition = self.decomposition.applyfunc(lambda x: 1/x)
            self.inv_covariance = self.solution_matrix.applyfunc(lambda x: 1/x)
        
        self.diffusion_call = lambdify(variable, self.diffusion, module)

        self.decomposition = lambdify(variable, self.decomposition, module)

        self.inv_decomposition = lambdify(variable, self.inv_decomposition, module)

        # If Sigma(0) != 0, this method has to be constructed in the SDE
        self.inv_covariance = lambdify(variable, self.inv_covariance, module)

        if module == 'jax':
            
            self.diffusion_call = jax.vmap(self.diffusion_call)

            self.decomposition = jax.vmap(self.decomposition)

            self.inv_decomposition = jax.vmap(self.inv_decomposition)

            self.inv_covariance = jax.vmap(self.inv_covariance)

    def __call__(self, time) -> Any:
        return self.diffusion_call(time)        

    def __repr__(self):
        return str(self.diffusion)
class SDE:
    
    def __init__(self, variable, drift, diffusion, diffusion_matrix, initial_variable_value = 0., max_variable_value = math.inf, module='jax', drift_integral_form = False, diffusion_integral_form = False, diffusion_integral_decomposition = 'cholesky', drift_diagonal_form = True, diffusion_diagonal_form = True, diffusion_matrix_diagonal_form = True):
    
        self.drift = DRIFT(variable, drift, initial_variable_value, module, drift_integral_form, drift_diagonal_form)
        self.diffusion = DIFFUSION(variable, diffusion, diffusion_matrix, initial_variable_value, module, diffusion_integral_form, diffusion_integral_decomposition, diffusion_diagonal_form, diffusion_matrix_diagonal_form)

        # USED FOR GENERATING TIME STEPS
        self.initial_variable_value = initial_variable_value
        self.max_variable_value = max_variable_value


    def sample(self, timestep, initial_data, key):
        """
        Sample a noisy datapoit using\\
        x(t)=mu(t) + A(t) z_t  |  z_t \~ N(0,1)\\
        which is the same as:\\
        x(t) \~ N( mu(t),Sigma(t) )
        """
        
        key, subkey = jax.random.split(key)
        
        z = jax.random.normal(subkey, initial_data.shape)

        mean = self.mean(timestep, initial_data)

        A = self.diffusion.decomposition(timestep) # decompose Sigma into A: Sigma(t)=A(t)@A(t).T

        if self.diffusion.diagonal_form:
            decomposition_product = A.squeeze(1) * z
        else:
            decomposition_product = batch_matrix_vec(A, z) 

        return mean + decomposition_product, z # x(t)=mu(t) + A(t) z_t, z_t

    def mean(self, timestep, initial_data):
        """
        timestep \in R\\
        initial_data \in R^(n)
        """

        exp_F = self.drift.solution_matrix(timestep)

        if self.drift.diagonal_form:
            mean = exp_F.squeeze(1) * initial_data
        else:
            mean = batch_matrix_vec(exp_F, initial_data)

        return mean

    def score(self, timestep, initial_data, data):
        """
        Returns the score:   Σ^(-1)(t) (x(t)-μ(t))
        """ 
        if self.diffusion.diagonal_form:
            return -self.diffusion.inv_covariance(timestep) * (data-self.mean(timestep, initial_data))
        else:
            return -self.diffusion.inv_covariance(timestep) @ (data-self.mean(timestep, initial_data))

    def step(self, model_output, timestep, data, key, dt, method: str = "euler/heun"):
        """
        Remove noise from image using the reverse SDE
        dx(t) = [ F(t)x(t) - L(t)L(t)^T score ] dt + L(t) db(t)
        """
        key, noise_key = jax.random.split(key) # IDEA: TRY SEMI DETERMINISTIC KEYING VS RANDOM
        # TODO (KLAUS): MAKE TIME AND X SAME SIZE
        def d(n_x, t, key):
            """
            Evaluate dx/dt at x,t\\
            TODO: Talk to klaus about the correctness of this
            """
            noise = jax.random.normal(key, n_x.shape) # TODO: ask klaus if each function eval should use a new random noise, or the same, it currently uses the same. Answer: Yes, they should

            Fx = self.mean(t, n_x) 
            # print("Fx =",Fx) # TODO (KLAUS): try print(f"{Fx=}") and observe magic
            L = self.diffusion(t)
            # print("L =",L)

            if self.diffusion.diagonal_form:
                L = L.squeeze(1)
                diffusion_term = - L * L * model_output + L * noise
            else:
                diffusion_term = - L @ L.T @ model_output + L @ noise

            dxdt = Fx + diffusion_term 
            print(f"{Fx=}, {L=}, {dxdt=}, {t=}") 
            print(self.diffusion.diffusion.subs(Symbol('t', nonnegative=True, real=True),1))
            # Our SDE: dx(t) = [ F(t)x(t) - L(t)L(t)^T score ] dt + L(t) db(t)

            # Yang Song dx = [ f(x,t)-g^2 score ] dt + g(t) dw
            # what he does:
            # prev_sample = sample - drift + diffusion**2 * model_output + diffusion * noise
            # f(x,t) = -drift
            # g(t) = diffusion
            # dw = noise
            

            # sample is x in x+h*f in Euler
            # the h from Euler comes through the diffusion part i think, im not certain

            return dxdt

        derivative = d(data, timestep, noise_key) # find dx/dt

        ### Euler ###
        dx = dt*derivative
        euler = data + dx

        # if method == "euler" or t == 0: # not nice to use if-statement in jax
        #     return x_n1
        return euler, key

        # TODO (ANDREAS): THE SECOND CALL TO d requires calling the model again. I understand why step correct exists.
        ### Runge kutta ###
        key, noise_key = jax.random.split(key)
        # elif method == "heun":        
        a = 1 # 1 = Heun method, 1/2 = midpoint method, 3/2 = Ralstons method
        ls = (1-1/(2*a))* dx # left side
        rs = dt/(2*a) * d(euler, timestep+a*dt, noise_key) # right side
        x_n2 = data + ls + rs

        return x_n2, key


        # self.mean(timestep, noisy_data) = F(t)*x(t)
        # L(t) = drift, through model call, look at pipeline


        # self.scheduler must have all functions we call on it
        pass

def sample(timestep, initial_data, key):
    key, subkey = jax.random.split(key)
        
    z = jax.random.normal(subkey, initial_data.shape)

    exp_F = jnp.exp(jnp.sin(timestep))

    mean = jax.vmap(lambda a, b: a * b)(exp_F.reshape(-1,1), initial_data)
    
    A = jnp.sqrt(timestep/2 - jnp.sin(timestep)*jnp.cos(timestep)/2)

    decomposition_product = jax.vmap(lambda a, b: a * b)(A, z)

    return mean + decomposition_product

if __name__ == "__main__":
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

    t = Symbol('t', nonnegative=True, real=True)
    key = jax.random.PRNGKey(0)
    timesteps = jnp.array([0.1,1,1.5]) # TODO, ask klaus if len(timesteps) =  batchsize, or why the things take multiple timesteps

    n = 1 # dims of problem

    x0 = jnp.ones((len(timesteps), n))

    #custom_vector = sample(timesteps, jnp.ones((len(timesteps), 435580)), key)


    F = Matrix.diag([sympy.cos(t)]*n)
    L = Matrix.diag([sympy.sin(t)]*n)
    Q = Matrix.eye(n)

    # Normal test
    sde = SDE(t, F, L, Q, drift_diagonal_form=False, diffusion_diagonal_form=False, diffusion_matrix_diagonal_form=False)
    normal_matrix = sde.sample(timesteps, x0, key)[0]

    # Integral test
    sde = SDE(t, F, L, Q, drift_integral_form=True, diffusion_integral_form=True, drift_diagonal_form=False, diffusion_diagonal_form=False, diffusion_matrix_diagonal_form=False)
    integral_matrix = sde.sample(timesteps, x0, key)[0]

    # Diagonal tests
    sde = SDE(t, F.diagonal(), L.diagonal(), Q.diagonal(), drift_diagonal_form=True, diffusion_diagonal_form=True, diffusion_matrix_diagonal_form=True)
    normal_vector = (sde.sample(timesteps, x0, key)[0])
    
    print((sde.sample(timesteps, jnp.ones((len(timesteps), 435580)), key)[0]).shape)
    #assert jnp.array_equal(normal_vector, custom_vector)

    sde = SDE(t, F.diagonal(), L, Q.diagonal(), drift_diagonal_form=True, diffusion_diagonal_form=False, diffusion_matrix_diagonal_form=True)
    normal_diag_matrix_0 = sde.sample(timesteps, x0, key)[0]

    sde = SDE(t, F.diagonal(), L.diagonal(), Q, drift_diagonal_form=True, diffusion_diagonal_form=True, diffusion_matrix_diagonal_form=False)
    normal_diag_matrix_1 = sde.sample(timesteps, x0, key)[0]

    # Diagonal Integral tests
    sde = SDE(t, F.diagonal(), L.diagonal(), Q.diagonal(), drift_integral_form=True, diffusion_integral_form=True, drift_diagonal_form=True, diffusion_diagonal_form=True, diffusion_matrix_diagonal_form=True)
    integral_vector = sde.sample(timesteps, x0, key)[0]
    print(integral_matrix.device(), integral_vector.device())
    assert jnp.array_equal(normal_matrix, normal_vector)
    assert jnp.array_equal(normal_vector, normal_diag_matrix_0)
    assert jnp.array_equal(normal_diag_matrix_0, normal_diag_matrix_1)
    print("Diagonal representation tests passed:)")

    assert jnp.array_equal(integral_matrix, integral_vector)
    print("Diagonal integral representation tests passed:)")
    print(normal_matrix.shape, normal_vector.shape, normal_diag_matrix_0.shape)

#    import timeit 
#    t0 = timeit.time.time() 
#    sde = SDE(t, Matrix([[1/(t+1)]*435580]), Matrix([[t]*435580]), Matrix([[1]*435580]), drift_diagonal_form=True, diffusion_diagonal_form=True, diffusion_matrix_diagonal_form=True)
#    t1 = timeit.time.time()
#    print(t1-t0)
#
#    t0 = timeit.time.time()
#    sde = SDE(t, Matrix([[1/(t+1)]*435580]), Matrix([[t]*435580]), Matrix([[1]*435580]), drift_integral_form=True, diffusion_integral_form=True, drift_diagonal_form=True, diffusion_diagonal_form=True, diffusion_matrix_diagonal_form=True)
#    t1 = timeit.time.time()
#    print(t1-t0)

    model_out = jnp.ones((len(timesteps), n))
    h = 0.05 # too big h results in nan, already at 0.1 this happens, The NANs appear from the diffusion term.
    prev_x = sde.step(timestep= timesteps, next_timestep=timesteps+h, noisy_x=x0, model_output = model_out, key = key, method = "huen")
    # print(prev_x)