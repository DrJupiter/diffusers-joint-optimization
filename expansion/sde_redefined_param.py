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
    
    def __init__(self, variable, parameters, drift, initial_value=0., module = 'jax', integral_form=False, diagonal=True):
        
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
        
            self.symbolic_solution_matrix =  (self.drift_int-self.drift_int.limit(variable, initial_value)).exp()
        else:
            # We can split up the exponential
            #self.solution_matrix = lambdify(variable, matrix_multiply_elementwise(self.drift_int.applyfunc(sympy.exp).transpose(),(-self.drift_int).applyfunc(sympy.exp).limit(variable,initial_value).transpose()), module)
            self.symbolic_solution_matrix = (self.drift_int-self.drift_int.limit(variable,initial_value)).applyfunc(sympy.exp)

        self.solution_matrix = lambdify([variable, parameters], self.symbolic_solution_matrix, module)

        self.drift_call = lambdify([variable,parameters], self.drift, module)

        if module == 'jax':
            self.drift_call = jax.vmap(self.drift_call, (0, None))
            self.solution_matrix = jax.vmap(self.solution_matrix, (0, None))
        
    def __repr__(self):
        return str(self.drift)

    def __call__(self, time, parameters, data) -> Any:
        if self.diagonal_form:
            return self.drift_call(time, parameters) * data
        else:
            return self.drift_call(time, parameters) @ data

class DIFFUSION:
    
    def __init__(self, variable, parameters, diffusion, diffusion_matrix=None, initial_value=0., module='jax', integral_form=False, integral_decomposition='cholesky', diagonal=True, diagonal_diffusion=True):
    
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

        # TODO (KLAUS) : SEE IF POSSIBLE TO REIMPLEMENT LIMIT, CAUSES PROBLEMS WITH MAX AND MIN FUNCTIONS ON SERIES                
        #self.solution_matrix = (self.diffusion_int-self.diffusion_int.limit(variable, initial_value, '+'))
        self.solution_matrix = (self.diffusion_int-self.diffusion_int.subs(variable, initial_value))
        
        if not self.diagonal_form:
            
            # TODO: ADD CHECK TO SEE IF THE SOLUTION IS VALID
            self.decomposition = self.solution_matrix.cholesky()
            self.inv_decomposition = self.decomposition.inv()
            self.inv_covariance = self.solution_matrix.inv()
        else:
            
            self.decomposition = self.solution_matrix.applyfunc(sympy.sqrt)
            self.inv_decomposition = self.decomposition.applyfunc(lambda x: 1/x)
            self.inv_covariance = self.solution_matrix.applyfunc(lambda x: 1/x)
        
        self.diffusion_call = lambdify([variable, parameters], self.diffusion, module)

        self.symbolic_decomposition = self.decomposition
        self.decomposition = lambdify([variable, parameters], self.decomposition, module)

        self.inv_decomposition = lambdify([variable, parameters], self.inv_decomposition, module)

        # If Sigma(0) != 0, this method has to be constructed in the SDE
        self.inv_covariance = lambdify([variable, parameters], self.inv_covariance, module)

        if module == 'jax':
            
            self.diffusion_call = jax.vmap(self.diffusion_call, (0, None))

            self.decomposition = jax.vmap(self.decomposition, (0, None))

            self.inv_decomposition = jax.vmap(self.inv_decomposition, (0, None))

            self.inv_covariance = jax.vmap(self.inv_covariance, (0, None))

    def __call__(self, time, parameters) -> Any:
        return self.diffusion_call(time, parameters)        

    def __repr__(self):
        return str(self.diffusion)

class SDE_PARAM:
    
    def __init__(self, variable, drift_parameters, diffusion_parameters, drift, diffusion, diffusion_matrix, initial_variable_value = 0., max_variable_value = math.inf, module='jax', model_target="epsilon", drift_integral_form = False, diffusion_integral_form = False, diffusion_integral_decomposition = 'cholesky', drift_diagonal_form = True, diffusion_diagonal_form = True, diffusion_matrix_diagonal_form = True):

        self.variable = variable
        self.drift_parameters = drift_parameters
        self.diffusion_parameters = diffusion_parameters 
        self.module = module

        self.drift = DRIFT(variable, drift_parameters, drift, initial_variable_value, module, drift_integral_form, drift_diagonal_form)
        self.diffusion = DIFFUSION(variable, diffusion_parameters, diffusion, diffusion_matrix, initial_variable_value, module, diffusion_integral_form, diffusion_integral_decomposition, diffusion_diagonal_form, diffusion_matrix_diagonal_form)

        # USED FOR GENERATING TIME STEPS
        self.initial_variable_value = initial_variable_value
        self.max_variable_value = max_variable_value
        self.model_target = model_target

        # batch_dim is one, because inputs will be vmapped or vectorized in another fashion.
        self.batch_dim = 1
        self.data_dim = sympy.symbols("data_dim", integer=True, nonnegative=True)

        self.symbolic_input = sympy.MatrixSymbol("X", self.batch_dim, self.data_dim)

        self.symbolic_noise = sympy.MatrixSymbol("z", self.batch_dim, self.data_dim)


    def lambdify_symbolic_functions(self, data_dimension):
        """
        Initialize The functions calculated symbolically for a specific data dimension,
        so they can be called within the specifc module
        """

        # CONCRETE DIMENSION FOR SYMBOLIC CALCULATIONS
        symbolic_sample = self.symbolic_sample().subs({self.data_dim: data_dimension})
        symbolic_input = self.symbolic_input.subs({self.data_dim: data_dimension})
        symbolic_noise = self.symbolic_noise.subs({self.data_dim: data_dimension})

        # SAMPLE
        self.lambdified_sample = lambdify([self.variable, self.symbolic_input.subs({self.data_dim:data_dimension}),self.symbolic_noise.subs({self.data_dim:data_dimension}), self.drift_parameters, self.diffusion_parameters],  symbolic_sample, self.module)

        # DERIVATIVE OF SAMPLE W.R.T DRIFT AND DIFFUSION 
        drift_derivative = symbolic_sample.diff(self.drift_parameters)
        diffusion_derivative = symbolic_sample.diff(self.diffusion_parameters)

        lambdified_drift_derivative = lambdify([self.variable, symbolic_input , symbolic_noise, self.drift_parameters, self.diffusion_parameters], drift_derivative, self.module)
        lambdified_diffusion_derivative = lambdify([self.variable, symbolic_input , symbolic_noise, self.drift_parameters, self.diffusion_parameters], diffusion_derivative, self.module)

        # CONVERT JACOBIANS TO NUMERATOR VECTOR LAYOUT
        if self.module == "jax":
            self.lambdified_drift_derivative = lambda t, data, noise, drift, diffusion: jnp.squeeze(lambdified_drift_derivative(t, data, noise, drift, diffusion)).T 
            self.lambdified_diffusion_derivative = lambda t, data, noise, drift, diffusion: jnp.squeeze(lambdified_diffusion_derivative(t, data, noise, drift, diffusion)).T 

        else:
            self.lambdified_drift_derivative = lambdified_drift_derivative
            self.lambdified_diffusion_derivative = lambdified_diffusion_derivative

            print("derivatives haven't been converted to numerator layout. Consider if this is relevant for your use case. Otherwise squeeze and transpose the output to achieve this.")

        if self.module == "jax":
            # TODO (KLAUS): HOPEFULLY WE CAN JUST USE THIS IN THE TORCH CASE, WE HAVE TO CHECK IF IT WILL ALLOW US TO LAMBIDIFY
            self.v_lambdified_sample = jax.vmap(self.lambdified_sample, (0, 0, 0, None, None)) 
            self.v_lambdified_drift_derivative = jax.vmap(self.lambdified_drift_derivative, (0, 0, 0, None, None))
            self.v_lambdified_diffusion_derivative = jax.vmap(self.lambdified_diffusion_derivative, (0, 0, 0, None, None))
    
    def symbolic_sample(self):
        """
        Sample a noisy datapoit using\\
        x(t)=mu(t) + A(t) z_t  |  z_t \~ N(0,1)\\
        which is the same as:\\
        x(t) \~ N( mu(t),Sigma(t) )
        """
        

        # TODO (KLAUS): CLASS SHOULD TAKE IN SAMPLE DIM? b, d
        #z = jax.random.normal(subkey, initial_data.shape)

        mean = self.symbolic_mean()

        A = self.diffusion.symbolic_decomposition # decompose Sigma into A: Sigma(t)=A(t)@A(t).T

        if self.diffusion.diagonal_form:
            decomposition_product = A.squeeze(1) * self.symbolic_noise
        else:
            decomposition_product = self.symbolic_noise @ A.T 

        return mean + decomposition_product # x(t)=mu(t) + A(t) z_t, z_t

    def symbolic_mean(self):
        """
        timestep \in R\\
        initial_data \in R^(n)
        """

        exp_F = self.drift.symbolic_solution_matrix

        if self.drift.diagonal_form:
            mean = exp_F.squeeze(1) * self.symbolic_input
        else:
            mean = self.symbolic_input @ exp_F.T 

        return mean
      

    def sample(self, timestep, drift_parameters, diffusion_parameters, initial_data, key):
        """
        Sample a noisy datapoit using\\
        x(t)=mu(t) + A(t) z_t  |  z_t \~ N(0,1)\\
        which is the same as:\\
        x(t) \~ N( mu(t),Sigma(t) )
        """
        
        key, subkey = jax.random.split(key)
        
        z = jax.random.normal(subkey, initial_data.shape)

        mean = self.mean(timestep, drift_parameters, initial_data)

        A = self.diffusion.decomposition(timestep, diffusion_parameters) # decompose Sigma into A: Sigma(t)=A(t)@A(t).T

        if self.diffusion.diagonal_form:
            decomposition_product = A.squeeze(1) * z
        else:
            decomposition_product = batch_matrix_vec(A, z) 

        return mean + decomposition_product, z # x(t)=mu(t) + A(t) z_t, z_t

    def mean(self, timestep, drift_parameters, initial_data):
        """
        timestep \in R\\
        initial_data \in R^(n)
        """

        exp_F = self.drift.solution_matrix(timestep, drift_parameters)

        if self.drift.diagonal_form:
            mean = exp_F.squeeze(1) * initial_data
        else:
            mean = batch_matrix_vec(exp_F, initial_data)

        return mean

    def score(self, timestep, drift_parameters, diffusion_parameters, initial_data, data):
        """
        Returns the score:   Σ^(-1)(t) (x(t)-μ(t))
        """ 
        if self.diffusion.diagonal_form:
            return -self.diffusion.inv_covariance(timestep, diffusion_parameters) * (data-self.mean(timestep, drift_parameters, initial_data))
        else:
            return -self.diffusion.inv_covariance(timestep, diffusion_parameters) @ (data-self.mean(timestep, drift_parameters, initial_data))

    def reverse_time_derivative(self, noisy_data, model_output, drift_parameters, diffusion_parameters, timestep, key):
        """
        Evaluate dx/dt at noisy_data,t\\
        TODO: Talk to klaus about the correctness of this (should be resolved)\\
        TODO: better name?
        """

        # generate random noise
        noise = jax.random.normal(key, noisy_data.shape) # TODO: ask klaus if each function eval should use a new random noise, or the same, it currently uses the same. Answer: Yes, they should

        # Find mean of noisy data at timestep t
        Fx = self.mean(timestep, drift_parameters, noisy_data) 

        # Find diffusion at timestep t
        L = self.diffusion(timestep, diffusion_parameters)

        # Finds core at timestep t based on the intial data and the noisy data
        if self.model_target == "epsilon":
            if self.diffusion.diagonal_form:
                score = - self.diffusion.inv_decomposition(timestep, diffusion_parameters).squeeze(1) * model_output
            else:
                score = -self.diffusion.inv_decomposition(timestep, diffusion_parameters) @ model_output

        elif self.model_target == "score":
            score = model_output

        else:
            raise ValueError(f"Unable to calculate score based on Model Target {self.model_target}")

        # use correct form baed on if its a digagonal form or a filled matrix
        if self.diffusion.diagonal_form: # diagonal
            L = L.squeeze(1)
            diffusion_term = - L * L * score + L * noise
        else: # filled
            diffusion_term = - L @ L.T @ score + L @ noise

        # F(t) x(t) +  ( L(t) L(t) score + L(t)*noise )
        dxdt = Fx + diffusion_term 

        #print(f"{L=}, {score=},{self.diffusion.inv_decomposition(t)=}")
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

    def step(self, model_output, timestep, drift_parameters, diffusion_parameters, noisy_data, key, dt):
        """
        Remove noise from image using the reverse SDE
        dx(t) = [ F(t)x(t) - L(t)L(t)^T score ] dt + L(t) db(t)
        """
        key, noise_key = jax.random.split(key) # IDEA: TRY SEMI DETERMINISTIC KEYING VS RANDOM
        

        # evaluate reverse SDE in the current datapoint and timestep
        dxdt = self.reverse_time_derivative(noisy_data, model_output, drift_parameters, diffusion_parameters, timestep, noise_key)

        ### Euler ###
        dx = dt*dxdt # h*f(t,x)
        euler_data = noisy_data + dx # x + h*f(t,x)

        return euler_data, dxdt, key

    def step_correct(self, model_output_euler_data, timestep, drift_parameters, diffusion_parameters, noisy_data, euler_data, dxdt, key, dt):
        """
        Perform Runge kutta correction update step.\\
        Should follow self.step() function using a seconday new model output based on the data returned in self.step().
        """

        ### Runge kutta ###
        key, noise_key = jax.random.split(key)

        # determine exact 2nd order method      
        a = 1 # 1 = Heun method, 1/2 = midpoint method, 3/2 = Ralstons method

        ## calculate    x + dt *(1-1/(2*a)) * dxdt   +   dt * 1/(2*a) * d(t+dt*a, new_data)

        # ls = dt *(1-1/(2*a)) * dxdt
        ls = dt * (1-1/(2*a))* dxdt # left side  --  dxdt = f(t,x), identical to what was used in the self.step() function

        # rs = dt * 1/(2*a) * d(t+dt*a, new_data)
        rs = dt/(2*a) * self.reverse_time_derivative(euler_data, model_output_euler_data, drift_parameters, diffusion_parameters, timestep+a*dt, noise_key) # right side

        # x + ls + rs
        corrected_step_data = noisy_data + ls + rs

        return corrected_step_data, key








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
    drift_param, diffusion_param = sympy.MatrixSymbol("Theta_F",3,1), sympy.MatrixSymbol("Theta_L",2,1) 
    x1,x2,x3,x4,x5 = sympy.symbols("x1 x2 x3 x4 x5", real=True)
    drift_param = Matrix([x1,x2,x3])
    diffusion_param = Matrix([x4,x5])

    v_drift_param =  jnp.array([1.,1.,1.])
    v_diffusion_param =  jnp.array([4.,4.])

    key = jax.random.PRNGKey(0)
    timesteps = jnp.array([0.1,0.4]) # TODO, ask klaus if len(timesteps) =  batchsize, or why the things take multiple timesteps

    n = 3 # dims of problem

    x0 = jnp.ones((len(timesteps), n))*1/2

    #custom_vector = sample(timesteps, jnp.ones((len(timesteps), 435580)), key)


    F = Matrix.diag([[sympy.cos(t*drift_param[1])*drift_param[0] + drift_param[2]],[t*drift_param[2]], [t]])
    #F = Matrix.diag([sympy.cos(t*drift_param[1])*drift_param[0] + drift_param[2]]*n)
    #L = Matrix.diag([sympy.sin(t)*diffusion_param[0]+diffusion_param[1]]*n)
    L = Matrix.diag([[sympy.sin(t)*diffusion_param[0]+diffusion_param[1]],[t * diffusion_param[1] ], [t]])
    Q = Matrix.eye(n)

    # Normal test
    sde = SDE_PARAM(t, drift_param, diffusion_param, F, L, Q, drift_diagonal_form=False, diffusion_diagonal_form=False, diffusion_matrix_diagonal_form=False)
    sde.lambdify_symbolic_functions(n)
    symbolic_sample = sde.symbolic_sample().subs({sde.data_dim: n})
    #print(symbolic_sample)
    #symbolic_sample = sympy.simplify(symbolic_sample)
    ss = lambdify([sde.variable, sde.symbolic_input.subs({sde.data_dim:n}),sde.symbolic_noise.subs({sde.data_dim:n}), sde.drift_parameters, sde.diffusion_parameters],  symbolic_sample, "jax")
    ss_derivate_drift = lambdify([sde.variable, sde.symbolic_input.subs({sde.data_dim:n}),sde.symbolic_noise.subs({sde.data_dim:n}), sde.drift_parameters, sde.diffusion_parameters],  sympy.simplify(symbolic_sample.diff(sde.drift_parameters)), "jax")
    num_ss_derivate_drift = lambda t, data, noise, drift, diffusion: jnp.squeeze(ss_derivate_drift(t, data, noise, drift, diffusion)).T
    key2, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, x0.shape)
    #print(ss(timesteps, x0, z, v_drift_param, v_diffusion_param))
    v_ss = jax.vmap(ss, (0,0, 0, None, None))
    v_ss_drift = jax.vmap(lambda t, data, noise, drift, diffusion: jnp.squeeze(ss_derivate_drift(t, data, noise, drift, diffusion)).T, (0,0, 0, None, None))
    out_v = v_ss(timesteps, x0, z, v_drift_param, v_diffusion_param)
    print(sde.v_lambdified_sample(timesteps, x0, z, v_drift_param, v_diffusion_param)) 
    print(out_v)
    
    normal_matrix = sde.sample(timesteps,v_drift_param, v_diffusion_param, x0, key)[0]
    print(normal_matrix)
    print(sde.v_lambdified_diffusion_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    print(sde.v_lambdified_drift_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    out_v_grad = v_ss_drift(timesteps, x0, z, v_drift_param, v_diffusion_param)
    print(out_v_grad)
    grad_sample = jax.jacobian(lambda t, drift_p, diff_p, data, key: sde.sample(t,drift_p,diff_p, data, key), (1,2))
    grads = grad_sample(timesteps, v_drift_param, v_diffusion_param, x0, key)
    print(grads[0][0])
    
    # Integral test
    sde = SDE_PARAM(t, drift_param, diffusion_param, F, L, Q, drift_integral_form=True, diffusion_integral_form=True, drift_diagonal_form=False, diffusion_diagonal_form=False, diffusion_matrix_diagonal_form=False)
    integral_matrix = sde.sample(timesteps,v_drift_param, v_diffusion_param, x0, key)[0]

    # Diagonal tests
    sde = SDE_PARAM(t, drift_param, diffusion_param, F.diagonal(), L.diagonal(), Q.diagonal(), drift_diagonal_form=True, diffusion_diagonal_form=True, diffusion_matrix_diagonal_form=True)
    normal_vector = (sde.sample(timesteps,v_drift_param, v_diffusion_param, x0, key)[0])
    

    sde = SDE_PARAM(t, drift_param, diffusion_param, F.diagonal(), L, Q.diagonal(), drift_diagonal_form=True, diffusion_diagonal_form=False, diffusion_matrix_diagonal_form=True)
    normal_diag_matrix_0 = sde.sample(timesteps,v_drift_param, v_diffusion_param, x0, key)[0]

    sde = SDE_PARAM(t, drift_param, diffusion_param, F.diagonal(), L.diagonal(), Q, drift_diagonal_form=True, diffusion_diagonal_form=True, diffusion_matrix_diagonal_form=False)
    normal_diag_matrix_1 = sde.sample(timesteps,v_drift_param, v_diffusion_param, x0, key)[0]

    # Diagonal Integral tests
    sde = SDE_PARAM(t, drift_param, diffusion_param, F.diagonal(), L.diagonal(), Q.diagonal(), drift_integral_form=True, diffusion_integral_form=True, drift_diagonal_form=True, diffusion_diagonal_form=True, diffusion_matrix_diagonal_form=True)
    integral_vector = sde.sample(timesteps,v_drift_param, v_diffusion_param, x0, key)[0]
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
    h = 0.05 # aka dt -- too big h results in nan, already at 0.1 this happens, The NANs appear from the diffusion term.
    prev_x, dxdt, key = sde.step(model_out, timesteps, v_drift_param, v_diffusion_param, noisy_data = x0, key=key, dt=h)
    print(f"Step test shape = {prev_x.shape}")

    model_out2 = jnp.ones((len(timesteps), n))*2
    new_x, key = sde.step_correct(model_out2, timesteps, v_drift_param, v_diffusion_param, noisy_data=x0, euler_data = prev_x, dxdt=dxdt, key=key, dt=h)
    print(f"Step correct test shape = {new_x.shape}")
    print("Step and Step correct tests run :)")