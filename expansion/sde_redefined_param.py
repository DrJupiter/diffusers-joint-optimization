from typing import Any
import jax
from jax import numpy as jnp
import sympy
from sympy import Matrix, Symbol, lambdify, matrix_multiply_elementwise
import math


batch_matrix_vec = jax.vmap(lambda M, v: M @ v)
"""
[B, N, K] @ [B, K]
"""
from enum import Enum, auto

def matrix_diagonal_product(matrix, diagonal):
    return Matrix(list(map(lambda x: x[0] * x[1], zip([matrix[:,i] for i in range(matrix.shape[0])],diagonal)))).reshape(*matrix.shape).transpose()

def simple_squeeze(arr):
    new_shape = tuple(dim for dim in arr.shape if dim != 1)
    # handle case of (1,1,...,1)
    if len(new_shape) == 0:
        new_shape = (1,)
    return arr.reshape(*new_shape)

class SDEDimension(Enum):
    SCALAR   = auto()
    DIAGONAL = auto()
    FULL     = auto()

class DRIFT:
    
    def __init__(self, variable, parameters, drift, initial_value=0., module = 'jax', integral_form=False, dimension=SDEDimension.SCALAR):
        
        #self.diagonal_form = diagonal
        self.dimension = dimension

        if integral_form:
            # if drift is given integral form
            self.drift_int = drift
            self.drift = sympy.diff(self.drift_int, variable)
        else:
            self.drift = drift
            self.drift_int = sympy.integrate(drift,variable)

        match dimension:
            case SDEDimension.FULL:
                assert self.drift @ self.drift_int == self.drift_int @ self.drift, "The drift must commute with it's integral"
                # This check is not neccesary for diagonal matrices, as they will always commute.
        
                self.symbolic_solution_matrix =  (self.drift_int-self.drift_int.limit(variable, initial_value)).exp()
            case SDEDimension.DIAGONAL:
                # We can split up the exponential
                #self.solution_matrix = lambdify(variable, matrix_multiply_elementwise(self.drift_int.applyfunc(sympy.exp).transpose(),(-self.drift_int).applyfunc(sympy.exp).limit(variable,initial_value).transpose()), module)
                self.symbolic_solution_matrix = (self.drift_int-self.drift_int.limit(variable,initial_value)).applyfunc(sympy.exp)
            case SDEDimension.SCALAR:

                self.symbolic_solution_matrix = sympy.exp(self.drift_int-self.drift_int.limit(variable,initial_value))

        self.solution_matrix = lambdify([variable, parameters], self.symbolic_solution_matrix, module)

        self.drift_call = lambdify([variable,parameters], self.drift, module)

        if module == 'jax':
            self.drift_call = jax.vmap(self.drift_call, (0, None))
            self.solution_matrix = jax.vmap(self.solution_matrix, (0, None))
        
    def __repr__(self):
        return str(self.drift)

    def __call__(self, time, parameters, data) -> Any:
        match self.dimension:
            case SDEDimension.FULL: 
                return self.drift_call(time, parameters) @ data
            case SDEDimension.DIAGONAL | SDEDimension.SCALAR:

                return self.drift_call(time, parameters) * data

class DIFFUSION:
    
    def __init__(self, variable, parameters, diffusion, diffusion_matrix=None, initial_value=0., module='jax', integral_form=False, integral_decomposition='cholesky', diffusion_dimension=SDEDimension.SCALAR, diffusion_matrix_dimension=SDEDimension.SCALAR):
    
        #self.diagonal_form = diagonal and diagonal_diffusion
        self.diffusion_dimension = diffusion_dimension
        self.diffusion_matrix_dimension = diffusion_matrix_dimension

        if integral_form:
            
            if diffusion_matrix is not None:
                print("Diffusion matricies supplied with integral forms are ignored")
            
            self.diffusion_int = diffusion
            diffusion_term = sympy.diff(self.diffusion_int, variable)
            if integral_decomposition == 'cholesky':
                
                # TODO (KLAUS): the diffusion matrix should be genereted based on its supplied dimension
                if diffusion_dimension == SDEDimension.FULL:
                #if not diagonal:
                    self.diffusion = diffusion_term.cholesky()
                    self.diffusion_matrix = Matrix.eye(self.diffusion.shape[0])
                elif diffusion_dimension == SDEDimension.DIAGONAL:
                    self.diffusion = diffusion_term.applyfunc(sympy.sqrt)
                    # TODO: ADD CHECK TO SEE IF THE SOLUTION IS VALID
                    self.diffusion_matrix = sympy.ones(*self.diffusion.shape)
                elif diffusion_dimension == SDEDimension.SCALAR:
                    self.diffusion = sympy.sqrt(diffusion_term)
                    self.diffusion_matrix = 1
                    
            elif integral_decomposition == 'ldl':
                if diffusion_dimension == SDEDimension.FULL:
                #if not diagonal:
                    self.diffusion, self.diffusion_matrix = diffusion_term.LDLdecomposition()
                    assert self.diffusion_matrix.diff(variable).is_zero_matrix, f"The diffusion matrix must not be dependent on {variable}"
                elif diffusion_dimension == SDEDimension.SCALAR | diffusion_dimension == SDEDimension.DIAGONAL:
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

            match (diffusion_dimension, diffusion_matrix_dimension):

                case (SDEDimension.FULL, SDEDimension.FULL):
                    self.diffusion_int = sympy.integrate(self.diffusion@(self.diffusion_matrix@self.diffusion.transpose()), variable)
                
                case (SDEDimension.DIAGONAL, SDEDimension.DIAGONAL):
                    self.diffusion_int = sympy.integrate(matrix_multiply_elementwise(self.diffusion, matrix_multiply_elementwise(self.diffusion_matrix, self.diffusion)), variable)

                case (SDEDimension.FULL, SDEDimension.DIAGONAL):
                    self.diffusion_int = sympy.integrate(matrix_diagonal_product(self.diffusion, self.diffusion_matrix) @ self.diffusion.transpose(), variable)
                
                case (SDEDimension.DIAGONAL, SDEDimension.FULL):
                    self.diffusion_int = sympy.integrate(matrix_diagonal_product(matrix_diagonal_product(self.diffusion_matrix, self.diffusion).transpose(), self.diffusion).transpose(), variable)

                case (SDEDimension.SCALAR, _):
                    # This case works for any scalar diffusion combination
                    #self.diffusion_int = sympy.integrate(sympy.HadamardProduct(sympy.HadamardProduct((self.diffusion_matrix), self.diffusion.T), self.diffusion), variable) 
                    self.diffusion_int = sympy.integrate(self.diffusion * self.diffusion * self.diffusion_matrix, variable) 

                
                case (SDEDimension.FULL, SDEDimension.SCALAR):
                    self.diffusion_int = sympy.integrate(self.diffusion * self.diffusion_matrix * self.diffusion.T, variable) 
                
                case (SDEDimension.DIAGONAL, SDEDimension.SCALAR):
                    self.diffusion_int = sympy.integrate(self.diffusion_matrix * sympy.HadamardProduct((self.diffusion ,  self.diffusion.T)), variable) 

                case (SDEDimension.SCALAR, SDEDimension.SCALAR):
                    self.diffusion_int = sympy.integrate(self.diffusion * self.diffusion_matrix * self.diffusion, variable)

        
        # TODO (KLAUS) : SEE IF POSSIBLE TO REIMPLEMENT LIMIT, CAUSES PROBLEMS WITH MAX AND MIN FUNCTIONS ON SERIES                
        #self.solution_matrix = (self.diffusion_int-self.diffusion_int.limit(variable, initial_value, '+'))
        self.solution_matrix = (self.diffusion_int-self.diffusion_int.subs(variable, initial_value))

        match (diffusion_dimension, diffusion_matrix_dimension):
            case (SDEDimension.FULL, _) | (_, SDEDimension.FULL):
        #if diffusion_dimension == SDEDimension.FULL or diffusion_matrix_dimension == SDEDimension.FULL: 
        #if not self.diagonal_form:
            
            # TODO (KLAUS): ADD CHECK TO SEE IF THE SOLUTION IS VALID
                self.decomposition = self.solution_matrix.cholesky()
                self.inv_decomposition = self.decomposition.inv()
                self.inv_covariance = self.solution_matrix.inv()
            case (SDEDimension.DIAGONAL, _) | (_, SDEDimension.DIAGONAL):
        #else:
            
                self.decomposition = self.solution_matrix.applyfunc(sympy.sqrt)
                self.inv_decomposition = self.decomposition.applyfunc(lambda x: 1/x)
                self.inv_covariance = self.solution_matrix.applyfunc(lambda x: 1/x)
            case (SDEDimension.SCALAR, SDEDimension.SCALAR):
                self.decomposition = sympy.sqrt(self.solution_matrix)
                self.inv_decomposition = 1/self.decomposition
                self.inv_covariance = 1/self.solution_matrix
        
        self.diffusion_call = lambdify([variable, parameters], self.diffusion, module)

        self.symbolic_decomposition = self.decomposition
        #print(f"{self.symbolic_decomposition.shape=}")

        self.decomposition = lambdify([variable, parameters], self.decomposition, module)

        self.symbolic_inv_decomposition = self.inv_decomposition
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
    
    def __init__(self, variable, drift_parameters, diffusion_parameters, drift, diffusion, diffusion_matrix, initial_variable_value = 0., max_variable_value = math.inf, module='jax', model_target="epsilon", drift_integral_form = False, diffusion_integral_form = False, diffusion_integral_decomposition = 'cholesky', drift_dimension = SDEDimension.DIAGONAL, diffusion_dimension = SDEDimension.DIAGONAL, diffusion_matrix_dimension = SDEDimension.SCALAR):

        self.variable = variable
        self.drift_parameters = drift_parameters
        self.diffusion_parameters = diffusion_parameters 
        self.module = module

        self.drift = DRIFT(variable, drift_parameters, drift, initial_variable_value, module, drift_integral_form, drift_dimension)
        self.diffusion = DIFFUSION(variable, diffusion_parameters, diffusion, diffusion_matrix, initial_variable_value, module, diffusion_integral_form, diffusion_integral_decomposition, diffusion_dimension, diffusion_matrix_dimension)

        # USED FOR GENERATING TIME STEPS
        self.initial_variable_value = initial_variable_value
        self.max_variable_value = max_variable_value
        self.model_target = model_target

        # batch_dim is one, because inputs will be vmapped or vectorized in another fashion.
        self.batch_dim = 1
        self.data_dim = sympy.symbols("data_dim", integer=True, nonnegative=True)
        #self.data_dim = 4

        self.symbolic_input = sympy.MatrixSymbol("X", self.batch_dim, self.data_dim)

        self.symbolic_noise = sympy.MatrixSymbol("z", self.batch_dim, self.data_dim)

        self.symbolic_model = sympy.MatrixSymbol("s", self.batch_dim, self.data_dim)

        self.symbolic_target = sympy.MatrixSymbol("e", self.batch_dim, self.data_dim)

    def lambdify_symbolic_functions(self, data_dimension):
        """
        Initialize The functions calculated symbolically for a specific data dimension,
        so they can be called within the specifc module
        """
        # CONCRETE DIMENSION FOR SYMBOLIC CALCULATIONS
        symbolic_sample = self.symbolic_sample().subs({self.data_dim: data_dimension})
        symbolic_reverse_time_derivative = sympy.simplify(self.symbolic_reverse_time_derivative().subs({self.data_dim: data_dimension}))


        symbolic_input = self.symbolic_input.subs({self.data_dim: data_dimension})
        symbolic_noise = self.symbolic_noise.subs({self.data_dim: data_dimension})


        # For some reason sympy's diff engine will fail if we don't do this:
        #match (self.drift.dimension, self.diffusion.diffusion_dimension, self.diffusion.diffusion_matrix_dimension):
        #    case (SDEDimension.FULL,SDEDimension.FULL,SDEDimension.FULL):
        #        symbolic_model = Matrix([sympy.symbols(f"m_{{0:{data_dimension}}}")])
        #        symbolic_target = Matrix([sympy.symbols(f"t_{{0:{data_dimension}}}")])
        #    case (_,_,_):
        #        symbolic_model = self.symbolic_model.subs({self.data_dim: data_dimension})
        #        symbolic_target = self.symbolic_target.subs({self.data_dim: data_dimension})

        symbolic_model = Matrix([sympy.symbols(f"m_{{0:{data_dimension}}}")])
        symbolic_target = Matrix([sympy.symbols(f"t_{{0:{data_dimension}}}")])

        symbolic_loss = self.symbolic_scaled_loss().subs({self.symbolic_model: symbolic_model, self.symbolic_target: symbolic_target}).subs({self.data_dim: data_dimension}).doit()
        # SCALED LOSS
        lambdified_scaled_loss = lambdify([self.variable, symbolic_target, symbolic_model, self.drift_parameters, self.diffusion_parameters], symbolic_loss, self.module)

        # DERIVATIVE OF NORMALIZING FACTORS w.r.t MODEL, DRIFT, DIFFUSION

        lambdified_scaled_loss_derivative_model = lambdify([self.variable, symbolic_target, symbolic_model, self.drift_parameters, self.diffusion_parameters], simple_squeeze(sympy.simplify(symbolic_loss.diff(symbolic_model))), self.module)
        lambdified_scaled_loss_derivative_drift = lambdify([self.variable, symbolic_target, symbolic_model, self.drift_parameters, self.diffusion_parameters], simple_squeeze(sympy.simplify(symbolic_loss.diff(self.drift_parameters))), self.module)
        lambdified_scaled_loss_derivative_diffusion = lambdify([self.variable, symbolic_target, symbolic_model, self.drift_parameters, self.diffusion_parameters], simple_squeeze(sympy.simplify(symbolic_loss.diff(self.diffusion_parameters))), self.module)

        symbolic_model = self.symbolic_model.subs({self.data_dim: data_dimension})
        symbolic_target = self.symbolic_target.subs({self.data_dim: data_dimension})
        print(f"{symbolic_sample.shape=}, {symbolic_loss.shape=}")



        # SAMPLE
        lambdified_sample = lambdify([self.variable, symbolic_input, symbolic_noise, self.drift_parameters, self.diffusion_parameters],  symbolic_sample, self.module)

        # DERIVATIVE OF SAMPLE W.R.T DRIFT AND DIFFUSION 
        drift_derivative = symbolic_sample.diff(self.drift_parameters)
        diffusion_derivative = symbolic_sample.diff(self.diffusion_parameters)

        lambdified_drift_derivative = lambdify([self.variable, symbolic_input , symbolic_noise, self.drift_parameters, self.diffusion_parameters], drift_derivative, self.module)
        lambdified_diffusion_derivative = lambdify([self.variable, symbolic_input , symbolic_noise, self.drift_parameters, self.diffusion_parameters], diffusion_derivative, self.module)

        # REVERSE TIME DERIVATIVE
        lambdified_reverse_time_derivative = lambdify([self.variable, symbolic_input, symbolic_noise, symbolic_model, self.drift_parameters, self.diffusion_parameters],  symbolic_reverse_time_derivative, self.module)
        
        # CONVERT JACOBIANS TO NUMERATOR VECTOR LAYOUT
        if self.module == "jax":
            # SAMPLE
            self.lambdified_sample = lambda t, data, noise, drift, diffusion: jnp.squeeze(lambdified_sample(t, data, noise, drift, diffusion))
            self.lambdified_drift_derivative = lambda t, data, noise, drift, diffusion: jnp.squeeze(lambdified_drift_derivative(t, data, noise, drift, diffusion)).T 
            self.lambdified_diffusion_derivative = lambda t, data, noise, drift, diffusion: jnp.squeeze(lambdified_diffusion_derivative(t, data, noise, drift, diffusion)).T 

            # REVERSE SAMPLE
            self.lambdified_reverse_time_derivative = lambda t, data, noise, model, drift, diffusion: jnp.squeeze(lambdified_reverse_time_derivative(t, data, noise, model, drift, diffusion))

            # NORMALIZE
            self.lambdified_scaled_loss = lambda t, target, model, drift, diffusion : jnp.squeeze(lambdified_scaled_loss(t, target, model, drift, diffusion))   
            self.lambdified_scaled_loss_derivative_model = lambda t, target, model, drift, diffusion : jnp.squeeze(lambdified_scaled_loss_derivative_model(t, target, model, drift, diffusion)).T   
            self.lambdified_scaled_loss_derivative_drift = lambda t, target, model, drift, diffusion : jnp.squeeze(lambdified_scaled_loss_derivative_drift(t, target, model, drift, diffusion)).T      
            self.lambdified_scaled_loss_derivative_diffusion = lambda t, target, model, drift, diffusion : jnp.squeeze(lambdified_scaled_loss_derivative_diffusion(t, target, model, drift, diffusion)).T      

        else:
            self.lambdified_sample = lambdified_sample
            self.lambdified_drift_derivative = lambdified_drift_derivative
            self.lambdified_diffusion_derivative = lambdified_diffusion_derivative
            print("derivatives haven't been converted to numerator layout. Consider if this is relevant for your use case. Otherwise squeeze and transpose the output to achieve this.")

            self.lambdified_reverse_time_derivative = lambdified_reverse_time_derivative


        if self.module == "jax":
            # TODO (KLAUS): HOPEFULLY WE CAN JUST USE THIS IN THE TORCH CASE, WE HAVE TO CHECK IF IT WILL ALLOW US TO LAMBIDIFY
            self.v_lambdified_sample = jax.vmap(self.lambdified_sample, (0, 0, 0, None, None)) 
            self.v_lambdified_drift_derivative = jax.vmap(self.lambdified_drift_derivative, (0, 0, 0, None, None))
            self.v_lambdified_diffusion_derivative = jax.vmap(self.lambdified_diffusion_derivative, (0, 0, 0, None, None))
            self.v_lambdified_reverse_time_derivative = jax.vmap(self.lambdified_reverse_time_derivative, (0, 0, 0, 0, None, None))

            # t, target, model, drift, diffusion
            self.v_lambdified_scaled_loss = jax.vmap(self.lambdified_scaled_loss, (0, 0, 0, None, None) ) 
            self.v_lambdified_scaled_loss_derivative_model = jax.vmap(self.lambdified_scaled_loss_derivative_model, (0, 0, 0, None, None)) 
            self.v_lambdified_scaled_loss_derivative_drift = jax.vmap(self.lambdified_scaled_loss_derivative_drift, (0, 0, 0, None, None)) 
            self.v_lambdified_scaled_loss_derivative_diffusion = jax.vmap(self.lambdified_scaled_loss_derivative_diffusion, (0, 0, 0, None, None)) 
            


    def symbolic_sample(self):
        """
        Sample a noisy datapoit using
        x(t)=mu(t) + A(t) z_t   z_t N(0,1)
        which is the same as:
        x(t) N( mu(t),Sigma(t) )
        """

        mean = self.symbolic_mean()

        A = self.diffusion.symbolic_decomposition # decompose Sigma into A: Sigma(t)=A(t)@A(t).T

        match (self.diffusion.diffusion_dimension, self.diffusion.diffusion_matrix_dimension):
            case (SDEDimension.FULL, _) | (_, SDEDimension.FULL):
                decomposition_product = self.symbolic_noise @ A.T 
            case (SDEDimension.DIAGONAL, _) | (_, SDEDimension.DIAGONAL):
                decomposition_product = sympy.HadamardProduct(A, self.symbolic_noise)
            case (SDEDimension.SCALAR, SDEDimension.SCALAR):
                #decomposition_product = sympy.tensorproduct(A  self.symbolic_noise)

                decomposition_product = A * self.symbolic_noise
                #decomposition_product = sympy.HadamardProduct(self.symbolic_noise, A)


        return mean + decomposition_product 


    def symbolic_scaled_loss(self):
        

        exp_F = self.drift.symbolic_solution_matrix
        inv_A = self.diffusion.symbolic_inv_decomposition

        difference = self.symbolic_target - self.symbolic_model
        # DIMENSION OF A
        match (self.diffusion.diffusion_dimension, self.diffusion.diffusion_matrix_dimension):
            case (SDEDimension.FULL, _) | (_, SDEDimension.FULL):
                inv_decomposition_dimension = SDEDimension.FULL
            case (SDEDimension.DIAGONAL, _) | (_, SDEDimension.DIAGONAL):
                inv_decomposition_dimension = SDEDimension.DIAGONAL
            case (SDEDimension.SCALAR, SDEDimension.SCALAR):
                inv_decomposition_dimension = SDEDimension.SCALAR
        
        match (self.drift.dimension, inv_decomposition_dimension):

            case (SDEDimension.SCALAR, _) | (_, SDEDimension.SCALAR):
                # Probably something here
                scale = exp_F * inv_A
                loss = (difference @ scale @ difference.T).expand() 
            case (SDEDimension.DIAGONAL, SDEDimension.DIAGONAL):
                scale = sympy.HadamardProduct(exp_F, inv_A)
                loss = sympy.MatMul(difference, sympy.HadamardProduct(difference, scale).T) 
            case (SDEDimension.FULL, SDEDimension.DIAGONAL):
                scale = matrix_diagonal_product(exp_F, inv_A)
                loss = difference @ scale @ difference.T
            case (SDEDimension.DIAGONAL, SDEDimension.FULL):
                scale = matrix_diagonal_product(inv_A, exp_F)
                loss = difference @ scale @ difference.T
            case (SDEDimension.FULL, SDEDimension.FULL):
                scale = exp_F @ inv_A
                loss = difference @ scale @ difference.T

        return loss
                
        # 
        #match (self.drift.dimension, inv_decomposition_dimension):

        #    case (SDEDimension.FULL, _) | (_, SDEDimension.FULL):
        #         
        #    case (SDEDimension.DIAGONAL, _) | (_, SDEDimension.DIAGONAL):
        #        
        #    case (SDEDimension.SCALAR, SDEDimension.SCALAR):
                
        

    def symbolic_mean(self):
        """
        timestep 
        initial_data R^(n)
        """

        exp_F = self.drift.symbolic_solution_matrix

        match self.drift.dimension:
            case SDEDimension.FULL:
                mean = self.symbolic_input @ exp_F.T
            case SDEDimension.DIAGONAL:
                mean = sympy.HadamardProduct(exp_F, self.symbolic_input)
            case SDEDimension.SCALAR:
                # TODO (KLAUS): CAN WE DO BETTER THAN TENSOR PRODUCT? -> Problem jax.dot is produced with *
                #mean = sympy.tensorproduct(exp_F , self.symbolic_input)
                mean = exp_F * self.symbolic_input 
                #mean = sympy.HadamardProduct(self.symbolic_input,exp_F) 

        return mean
      

    def score(self, timestep, drift_parameters, diffusion_parameters, initial_data, data):
        """
        Returns the score:   Σ^(-1)(t) (x(t)-μ(t))
        """ 
        if self.diffusion.diagonal_form:
            return -self.diffusion.inv_covariance(timestep, diffusion_parameters) * (data-self.mean(timestep, drift_parameters, initial_data))
        else:
            return -self.diffusion.inv_covariance(timestep, diffusion_parameters) @ (data-self.mean(timestep, drift_parameters, initial_data))
#t, data, noise, drift, diffusion
    def symbolic_reverse_time_derivative(self):

        if self.model_target == "epsilon":
            match (self.diffusion.diffusion_dimension, self.diffusion.diffusion_matrix_dimension):
                case (SDEDimension.FULL, _) | (_, SDEDimension.FULL):
                    score = -  self.symbolic_model @ self.diffusion.symbolic_inv_decomposition.T
                case (SDEDimension.DIAGONAL, _) | (_, SDEDimension.DIAGONAL):
                    score = - sympy.HadamardProduct(self.diffusion.symbolic_inv_decomposition, self.symbolic_model)
                case (SDEDimension.SCALAR, SDEDimension.SCALAR):
                    score = - self.diffusion.symbolic_inv_decomposition * self.symbolic_model
        elif self.model_target == "score":
            score = self.symbolic_model
        else:
            raise ValueError(f"Unable to calculate score based on Model Target {self.model_target}")

        match self.diffusion.diffusion_dimension:
            case SDEDimension.FULL:
                diffusion_term = - score @ self.diffusion.diffusion @ self.diffusion.diffusion.T  + self.symbolic_noise @ self.diffusion.diffusion.T
            case SDEDimension.DIAGONAL:
                diffusion_term = - sympy.HadamardProduct(sympy.HadamardProduct(self.diffusion.diffusion, self.diffusion.diffusion), score) + sympy.HadamardProduct(self.diffusion.diffusion, self.symbolic_noise) 
            case SDEDimension.SCALAR:
                diffusion_term = - (self.diffusion.diffusion * self.diffusion.diffusion) * score + self.diffusion.diffusion * self.symbolic_noise

        return self.symbolic_mean() + diffusion_term 

    def step(self, data, reverse_time_derivative, dt):
        return data + dt * reverse_time_derivative

#    def step(self, model_output, timestep, drift_parameters, diffusion_parameters, noisy_data, key, dt):
#        """
#        Remove noise from image using the reverse SDE
#        dx(t) = [ F(t)x(t) - L(t)L(t)^T score ] dt + L(t) db(t)
#        """
#        key, noise_key = jax.random.split(key) # IDEA: TRY SEMI DETERMINISTIC KEYING VS RANDOM
#        
#
#        # evaluate reverse SDE in the current datapoint and timestep
#        dxdt = self.reverse_time_derivative(noisy_data, model_output, drift_parameters, diffusion_parameters, timestep, noise_key)
#
#        ### Euler ###
#        dx = dt*dxdt # h*f(t,x)
#        euler_data = noisy_data + dx # x + h*f(t,x)
#
#        return euler_data, dxdt, key

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
    from jax import config
    config.update("jax_enable_x64", True)
    sympy.init_printing(use_unicode=True, use_latex=True)
    t = Symbol('t', nonnegative=True, real=True)
    #drift_param, diffusion_param = sympy.MatrixSymbol("Theta_F",3,1), sympy.MatrixSymbol("Theta_L",2,1) 
    x1,x2,x3,x4,x5 = sympy.symbols("x1 x2 x3 x4 x5", real=True)

    drift_param = Matrix([x1])
    #diffusion_param = Matrix([x4,x5])
    diffusion_param = Matrix([x4])
    #drift_param = diffusion_param = sympy.symbols("∅", real=True)

    v_drift_param =  jnp.array([1.])
    v_diffusion_param =  jnp.array([4.])
    #v_drift_param = v_diffusion_param = jnp.array([None])

    key = jax.random.PRNGKey(0)
    timesteps = jnp.array([0.1]) # TODO, ask klaus if len(timesteps) =  batchsize, or why the things take multiple timesteps
    n = 4 # dims of problem

    x0 = jnp.ones((len(timesteps), n))*1/2
    z = jax.random.normal(key, x0.shape)
    print(z)
    model_output = jnp.ones_like(x0)

    #custom_vector = sample(timesteps, jnp.ones((len(timesteps), 435580)), key)

    F = Matrix.diag([sympy.cos(t*drift_param[0])]*n)
    L = Matrix.diag([sympy.sin(t)*diffusion_param[0]]*n) 
    Q = Matrix.eye(n)

    drift_dimension = SDEDimension.FULL
    drift_dimension = SDEDimension.DIAGONAL
    drift_dimension = SDEDimension.SCALAR


    S_F = sympy.cos(t*drift_param[0])
    S_L = sympy.sin(t)*diffusion_param[0]
    S_Q = 1 

    #print(F.shape, S_F.shape, S_L.shape, S_Q.shape)
    outputs = []
    integral_outputs = []
    #----

    diffusion_derivatives = []
    integral_diffusion_derivatives = []
    #----

    drift_derivatives = []
    integral_drift_derivatives = []
    #----

    reverse_time_derivatives = [] 
    integral_reverse_time_derivatives = [] 
    #----

    scaled_loss = []
    scaled_loss_model_derivative = []
    scaled_loss_drift_derivative = []
    scaled_loss_diffusion_derivative = []

    integral_scaled_loss = []
    integral_scaled_loss_model_derivative = []
    integral_scaled_loss_drift_derivative = []
    integral_scaled_loss_diffusion_derivative = []
    #----

    sde = SDE_PARAM(t, drift_param, diffusion_param, S_F, S_L, S_Q, drift_dimension=SDEDimension.SCALAR, diffusion_dimension=SDEDimension.SCALAR, diffusion_matrix_dimension=SDEDimension.SCALAR)
    sde.lambdify_symbolic_functions(n)
    #sde.lambdified_sample(timesteps[0],x0[0],z[0],v_drift_param, v_diffusion_param)

    outputs.append(sde.v_lambdified_sample(timesteps, x0, z, v_drift_param, v_diffusion_param) )
    diffusion_derivatives.append(sde.v_lambdified_diffusion_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    drift_derivatives.append(sde.v_lambdified_drift_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    reverse_time_derivatives.append(sde.v_lambdified_reverse_time_derivative(timesteps, x0, z, model_output, v_drift_param, v_diffusion_param))

    scaled_loss.append(sde.v_lambdified_scaled_loss(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    scaled_loss_model_derivative.append(sde.v_lambdified_scaled_loss_derivative_model(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    scaled_loss_drift_derivative.append(sde.v_lambdified_scaled_loss_derivative_drift(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    scaled_loss_diffusion_derivative.append(sde.v_lambdified_scaled_loss_derivative_diffusion(timesteps, z, model_output, v_drift_param, v_diffusion_param))

    sde = SDE_PARAM(t, drift_param, diffusion_param, F, L, Q, drift_dimension=SDEDimension.FULL, diffusion_dimension=SDEDimension.FULL, diffusion_matrix_dimension=SDEDimension.FULL)
    sde.lambdify_symbolic_functions(n)

    outputs.append(sde.v_lambdified_sample(timesteps, x0, z, v_drift_param, v_diffusion_param) )
    diffusion_derivatives.append(sde.v_lambdified_diffusion_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    drift_derivatives.append(sde.v_lambdified_drift_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    reverse_time_derivatives.append(sde.v_lambdified_reverse_time_derivative(timesteps, x0, z, model_output, v_drift_param, v_diffusion_param))


    scaled_loss.append(sde.v_lambdified_scaled_loss(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    scaled_loss_model_derivative.append(sde.v_lambdified_scaled_loss_derivative_model(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    scaled_loss_drift_derivative.append(sde.v_lambdified_scaled_loss_derivative_drift(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    scaled_loss_diffusion_derivative.append(sde.v_lambdified_scaled_loss_derivative_diffusion(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    
    # Integral test
    sde = SDE_PARAM(t, drift_param, diffusion_param, F, L, Q, drift_dimension=SDEDimension.FULL, diffusion_dimension=SDEDimension.FULL, diffusion_matrix_dimension=SDEDimension.FULL,
    drift_integral_form=True, diffusion_integral_form=True)
    sde.lambdify_symbolic_functions(n)
    integral_outputs.append(sde.v_lambdified_sample(timesteps, x0, z, v_drift_param, v_diffusion_param) )
    integral_diffusion_derivatives.append(sde.v_lambdified_diffusion_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    integral_drift_derivatives.append(sde.v_lambdified_drift_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    integral_reverse_time_derivatives.append(sde.v_lambdified_reverse_time_derivative(timesteps, x0, z, model_output, v_drift_param, v_diffusion_param))


    integral_scaled_loss.append(sde.v_lambdified_scaled_loss(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    integral_scaled_loss_model_derivative.append(sde.v_lambdified_scaled_loss_derivative_model(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    integral_scaled_loss_drift_derivative.append(sde.v_lambdified_scaled_loss_derivative_drift(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    integral_scaled_loss_diffusion_derivative.append(sde.v_lambdified_scaled_loss_derivative_diffusion(timesteps, z, model_output, v_drift_param, v_diffusion_param))

    # Diagonal tests
    sde = SDE_PARAM(t, drift_param, diffusion_param, F.diagonal(), L.diagonal(), Q.diagonal(), drift_dimension=SDEDimension.DIAGONAL, diffusion_dimension=SDEDimension.DIAGONAL, diffusion_matrix_dimension=SDEDimension.DIAGONAL)
    sde.lambdify_symbolic_functions(n)
    outputs.append(sde.v_lambdified_sample(timesteps, x0, z, v_drift_param, v_diffusion_param) )
    diffusion_derivatives.append(sde.v_lambdified_diffusion_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    drift_derivatives.append(sde.v_lambdified_drift_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    reverse_time_derivatives.append(sde.v_lambdified_reverse_time_derivative(timesteps, x0, z, model_output, v_drift_param, v_diffusion_param))
    

    scaled_loss.append(sde.v_lambdified_scaled_loss(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    scaled_loss_model_derivative.append(sde.v_lambdified_scaled_loss_derivative_model(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    scaled_loss_drift_derivative.append(sde.v_lambdified_scaled_loss_derivative_drift(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    scaled_loss_diffusion_derivative.append(sde.v_lambdified_scaled_loss_derivative_diffusion(timesteps, z, model_output, v_drift_param, v_diffusion_param))

    sde = SDE_PARAM(t, drift_param, diffusion_param, F.diagonal(), L, Q.diagonal(),drift_dimension=SDEDimension.DIAGONAL, diffusion_dimension=SDEDimension.FULL, diffusion_matrix_dimension=SDEDimension.DIAGONAL)
    sde.lambdify_symbolic_functions(n)

    outputs.append(sde.v_lambdified_sample(timesteps, x0, z, v_drift_param, v_diffusion_param) )
    diffusion_derivatives.append(sde.v_lambdified_diffusion_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    drift_derivatives.append(sde.v_lambdified_drift_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    reverse_time_derivatives.append(sde.v_lambdified_reverse_time_derivative(timesteps, x0, z, model_output, v_drift_param, v_diffusion_param))


    scaled_loss.append(sde.v_lambdified_scaled_loss(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    scaled_loss_model_derivative.append(sde.v_lambdified_scaled_loss_derivative_model(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    scaled_loss_drift_derivative.append(sde.v_lambdified_scaled_loss_derivative_drift(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    scaled_loss_diffusion_derivative.append(sde.v_lambdified_scaled_loss_derivative_diffusion(timesteps, z, model_output, v_drift_param, v_diffusion_param))


    sde = SDE_PARAM(t, drift_param, diffusion_param, F.diagonal(), L.diagonal(), Q,drift_dimension=SDEDimension.DIAGONAL, diffusion_dimension=SDEDimension.DIAGONAL, diffusion_matrix_dimension=SDEDimension.FULL)
    sde.lambdify_symbolic_functions(n)

    outputs.append(sde.v_lambdified_sample(timesteps, x0, z, v_drift_param, v_diffusion_param) )
    diffusion_derivatives.append(sde.v_lambdified_diffusion_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    drift_derivatives.append(sde.v_lambdified_drift_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    reverse_time_derivatives.append(sde.v_lambdified_reverse_time_derivative(timesteps, x0, z, model_output, v_drift_param, v_diffusion_param))


    scaled_loss.append(sde.v_lambdified_scaled_loss(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    scaled_loss_model_derivative.append(sde.v_lambdified_scaled_loss_derivative_model(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    scaled_loss_drift_derivative.append(sde.v_lambdified_scaled_loss_derivative_drift(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    scaled_loss_diffusion_derivative.append(sde.v_lambdified_scaled_loss_derivative_diffusion(timesteps, z, model_output, v_drift_param, v_diffusion_param))


    # Diagonal Integral tests
    sde = SDE_PARAM(t, drift_param, diffusion_param, F.diagonal(), L.diagonal(), Q.diagonal(), drift_dimension=SDEDimension.DIAGONAL, diffusion_dimension=SDEDimension.DIAGONAL, diffusion_matrix_dimension=SDEDimension.DIAGONAL, diffusion_integral_form=True, drift_integral_form=True)

    sde.lambdify_symbolic_functions(n)
    integral_outputs.append(sde.v_lambdified_sample(timesteps, x0, z, v_drift_param, v_diffusion_param) )
    integral_diffusion_derivatives.append(sde.v_lambdified_diffusion_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    integral_drift_derivatives.append(sde.v_lambdified_drift_derivative(timesteps, x0, z, v_drift_param, v_diffusion_param))
    integral_reverse_time_derivatives.append(sde.v_lambdified_reverse_time_derivative(timesteps, x0, z, model_output, v_drift_param, v_diffusion_param))


    integral_scaled_loss.append(sde.v_lambdified_scaled_loss(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    integral_scaled_loss_model_derivative.append(sde.v_lambdified_scaled_loss_derivative_model(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    integral_scaled_loss_drift_derivative.append(sde.v_lambdified_scaled_loss_derivative_drift(timesteps, z, model_output, v_drift_param, v_diffusion_param))
    integral_scaled_loss_diffusion_derivative.append(sde.v_lambdified_scaled_loss_derivative_diffusion(timesteps, z, model_output, v_drift_param, v_diffusion_param))

    from test.utils import test_arrays_equal
    test_arrays_equal(outputs, "Regular") 
    test_arrays_equal(integral_outputs, "Integral") 
    test_arrays_equal(diffusion_derivatives, "Regular Diffusion Derivative") 
    test_arrays_equal(drift_derivatives, "Regular Drift Derivative") 
    test_arrays_equal(reverse_time_derivatives, "Regular Reverse Time Derivative") 
    test_arrays_equal(integral_diffusion_derivatives, "Integral Diffusion Derivative") 
    test_arrays_equal(integral_drift_derivatives, "Integral Drift Derivative") 
    test_arrays_equal(integral_reverse_time_derivatives, "Integral Reverse Time Derivative") 

    test_arrays_equal(scaled_loss, "Scaled Loss")
    test_arrays_equal(scaled_loss_model_derivative ,"Scaled Loss Model Derivative") 
    test_arrays_equal(scaled_loss_drift_derivative, "Scaled Loss Drift Derivative")
    test_arrays_equal(scaled_loss_diffusion_derivative, "Scaled Loss Diffusion Derivative")

    test_arrays_equal(integral_scaled_loss, "Integral Scaled Loss")
    test_arrays_equal(integral_scaled_loss_model_derivative, "Integral Scaled Loss Model Derivative")
    test_arrays_equal(integral_scaled_loss_drift_derivative, "Integral Scaled Loss Drift Derivative")
    test_arrays_equal(integral_scaled_loss_diffusion_derivative, "Integral Scaled Loss Diffusion Derivative")