# Structure of Yang Song SDE VE Scheduler
## How it flows together

### Classes:

The main class `FlaxScoreSdeVeScheduler` inherits from `FlaxSchedulerMixin`. 
	`FlaxSchedulerMixin` inherits from `PushToHubMixin`

The inheritance results in the scheduler being able to upload to Huggingface (`PushToHubMixin`),
and to save and load scheduler settings (`FlaxSchedulerMixin`)

Two other classes are needed:
- `ScoreSdeVeSchedulerState` the class that defines the state
- `FlaxSdeVeOutput` The class the defines the output dict in place of a tuple. As such it is not important.


### Functions:

The two main functions are:
	`step_pred` and `step_correct`

These two functions share input and output.

`step_pred` predicts the next time step `scheduler.timesteps[t-1]` using the reverse SDE.
- `get_adjacent_sigma`, to get Sigma for (t-1)

`step_correct` corrects the prediction through multiple steps (can be seen below the code example).
- doesn't call any other functions in the class.

In the case of Jax another function `create_state` is also needed.

`create_state` calls
- the Class `ScoreSdeVeSchedulerState` with the function `.create()` to create the *state*.
- It then uses `set_sigmas` to alter Sigma in the *state*.

`set_sigmas` calls
- the function `set_timesteps` which alters the timestep in the *state*.
- `set_sigmas` uses this to update Sigma in the `state`

### Math
The main math in `step_pred` is:
$$diffusion = \sqrt{sigma_t^2 - sigma_{t-1}^2}$$
$$drift = -diffusion^2 \cdot \text{model\_output}$$
$$\mu_{\text{prev sample}} = sample - drift$$
$$\text{prev\_sample} = \mu_{\text{prev sample}} + diffusion \cdot noise$$

Where 
- the *sample* is the "image" we are currently working with
- *noise* is just normal
- *model_out* is the output of the model, aka the score ((?))

#### `step_correct` and `step_pred` use
Code taken from `src\diffusers\pipelines\score_sde_ve\pipeline_score_sde_ve.py`
```Python
class ScoreSdeVePipeline(DiffusionPipeline)
	... (more stuff) ...
	def __call__(
		... (more stuff) ...
		# correction step
		for _ in range(self.scheduler.config.correct_steps):
			model_output = self.unet(sample, sigma_t).sample
			sample = self.scheduler.step_correct(model_output, sample, generator=generator).prev_sample
		
		# prediction step
		model_output = model(sample, sigma_t).sample
		output = self.scheduler.step_pred(model_output, t, sample, generator=generator)
```
It shows that multiple `step_correct` are run for each `step_pred` as expected



## Main Class
### `ScoreSdeVeScheduler` or `FlaxScoreSdeVeScheduler`


#### Class functions:
- `__init__`
	- Description: "Initialises variables"
		- `num_train_timesteps`: int = 2000,
		- `snr`: float = 0.15,
		- `sigma_min`: float = 0.01,
		- `sigma_max`: float = 1348.0,
		- `sampling_eps`: float = 1e-5,
		- `correct_steps`: int = 1,

- `set_sigmas`    (support function (Jax it is for `create_state`)
	- Description: "Updates the state by: Setting the noise scales used for the diffusion chain. *Supporting function to be run before inference*. The sigmas control the weight of the `drift` and `diffusion` components of sample update."
	- Input: (`state`: `ScoreSdeVeSchedulerState`, `num_inference_steps`:`int`, `sigma_min`:`float`, `sigma_max`:`float`, `sampling_eps`:`float`)
	- Returns: `ScoreSdeVeSchedulerState` (same class as state)
		- NOTE: Torch version updates `__init__` instead of returning
	- Uses: `create_state` (in jax),     `__init__` (in torch)

- `set_timesteps`
	- Description: "Updates timesteps in state." 
		- Supporting function to be run before inference
	- Input: 
		- `state`: `ScoreSdeVeSchedulerState`, 
		- `num_inference_steps`: `int`, 
		- `shape`: `Tuple`, 
		- `sampling_eps`: `float`
	- Returns: 
		- `state`: `ScoreSdeVeSchedulerState` (In jax)
		- None (in torch), updates `__init__`
	- Uses: `set_sigmas`

- `get_adjacent_sigma`    (support function for `step_pred`)
	- Returns Sigma(timestep-1), but everywhere where $timestep_i$ == 0, it sets the value in $Sigma_i$ = 0.
	- Uses: `step_pred` 

- `step_pred`
	- Description: "Predict previous timestep using reverse SDE. Core function to propagate the diffusion process from the learned model outputs (most often the predicted noise)."
	- Input: 
		- `state`:`ScoreSdeVeSchedulerState`,  Current state
		- `model_output`: `jnp.ndarray`,  direct output from learned diffusion model
		- `timestep`: `int`,  current discrete timestep in the diffusion chain
		- `sample`: `jnp.ndarray`,  current instance of sample being created by diffusion process
		- `key`:` random.KeyArray`, 
		- `return_dict`:` bool`,  Determines if class or tuple is returned
	- Returns: `FlaxSdeVeOutput`(dict) or tuple consisting of
		- `state`: `ScoreSdeVeSchedulerState`
		- `prev_sample`: `jnp.ndarray`
		- `prev_sample_mean`: `Optional[jnp.ndarray]`

- `step_correct` (input and returns identical to `step_pred`)
	- Description: "Correct the predicted sample based on the output `model_output` of the network. This is often run repeatedly after making the prediction for the previous timestep"
	- Input: 
		- `state`:`ScoreSdeVeSchedulerState`,  Current state
		- `model_output`: `jnp.ndarray`,  direct output from learned diffusion model
		- `timestep`: `int`,  current discrete timestep in the diffusion chain
		- `sample`: `jnp.ndarray`,  current instance of sample being created by diffusion process
		- `key`:` random.KeyArray`, 
		- `return_dict`:` bool`,  Determines if class or tuple is returned
	- Returns: `FlaxSdeVeOutput`(dict) or tuple consisting of
		- `state`: `ScoreSdeVeSchedulerState`
		- `prev_sample`: `jnp.ndarray`
		- `prev_sample_mean`: `Optional[jnp.ndarray]`

- `__len__`
	- Description: 
	- Input: 
	- Returns: 

##### Jax only:
- `has_state`
	- Returns True
- `create_state`
	- Description: "Creates state using `ScoreSdeVeSchedulerState.create()` and passing it through `set_sigmas` (see `ScoreSdeVeSchedulerState` class for more info) "
	- Input: None
	- Returns: `ScoreSdeVeSchedulerState`

##### Torch only
- `scale_model_input`
- `add_noise`


## Supporting classes
### `ScoreSdeVeSchedulerState` (state)
Class that stores 3 variables:
1. `timesteps`: `Optional[jnp.ndarray]` = None
2. `discrete_sigmas`: `Optional[jnp.ndarray] `= None
3. `sigmas`: `Optional[jnp.ndarray]` = None

### `FlaxSdeVeOutput`
Class that stores 3 variables:
1. `state`: `ScoreSdeVeSchedulerState`
2. `prev_sample`: `jnp.ndarray`
3. `prev_sample_mean`: `Optional[jnp.ndarray]` = None



## Inheritance classes

### `PushToHubMixin`

This is the very deepest inheritance class.
It has the functions:
- `_upload_folder`
	Uploads all files in `working_dir` to `repo_id`.
- `push_to_hub`
	Upload model, scheduler, or pipeline files to the 🤗 Hugging Face Hub

The class can be found at: `src\diffusers\utils\hub_utils.py` (identical for Flax and torch)

Neither of these need be changed. Therefore we quickly skip over it.

The important class that inherits from this class is:
### `SchedulerMixin` or `FlaxSchedulerMixin`

Mixin containing common functions for the schedulers

Main purpose is saving and loading and getting 

Class methods:
- `from_pretrained`
	- Description: "Instantiate a Scheduler class from a pre-defined JSON-file as saved by `save_pretrained` function"
	- Input: `save_directory` type:  (`str` or `os.PathLike`)
	- Output: 
		- scheduler   type: 
		- state  type:
			- Based on scheduler using `sheduler.create_state()`

- `compatibles`
	- Description: "Returns all schedulers that are compatible with this scheduler"
	- Returns: `List[SchedulerMixin]`

[What it means for schedulers to be compatible is a different matter].

Class functions:
- `save_pretrained`
	- Description: "Save a scheduler configuration object to the directory `save_directory`, so that it can be re-loaded using the `.from_pretrained` class method"
	- Input: `save_directory` type:  (`str` or `os.PathLike`)
	- Output: None








# The different SDE/ODE schedulers

Here we will present all schedulers based on SDE's or ODE's for ease of comparison.
This will make it easier to write our own scheduler.

## SDE VE flax

### How it flows together

#### Classes:

The main class `FlaxScoreSdeVeScheduler` inherits from `FlaxSchedulerMixin`. 
	`FlaxSchedulerMixin` inherits from `PushToHubMixin`

The inheritance results in the scheduler being able to upload to Huggingface (`PushToHubMixin`),
and to save and load scheduler settings (`FlaxSchedulerMixin`)

Two other classes are needed:
- `ScoreSdeVeSchedulerState` the class that defines the state
- `FlaxSdeVeOutput` The class the defines the output dict in place of a tuple. As such it is not important.


#### Functions:

The two main functions are:
	`step_pred` and `step_correct`

These two functions share input and output.

`step_pred` predicts the next time step `scheduler.timesteps[t-1]` using the reverse SDE.
- `get_adjacent_sigma`, to get Sigma for (t-1)

`step_correct` corrects the prediction through multiple steps (can be seen below the code example).
- doesn't call any other functions in the class.

In the case of Jax another function `create_state` is also needed.

`create_state` calls
- the Class `ScoreSdeVeSchedulerState` with the function `.create()` to create the *state*.
- It then uses `set_sigmas` to alter Sigma in the *state*.

`set_sigmas` calls
- the function `set_timesteps` which alters the timestep in the *state*.
- `set_sigmas` uses this to update Sigma in the `state`


## DPM multistep flax

### Idea behind model

In this work, we propose an exact formulation of the solution of diffusion ODEs.
The formulation analytically computes the linear part of the solution, rather than leaving all terms to black-box ODE solvers as adopted in previous works
By applying change-of-variable, the solution can be equivalently simplified to an exponentially weighted integral of the neural network.
DPM-Solver can generate high-quality samples in only 10 to 20 function evaluations on various datasets
20-steps yield FID 2.87

So the sampling method used here is very different from what others do.

### Achievements

- FID 2.87 for 20 NN evals
- Only 20 NN evals from noise to img
- Uses other computations in addition to NN to sample
### How it flows together

#### Classes:

The main class `FlaxDPMSolverMultistepScheduler` inherits from `FlaxSchedulerMixin` and `ConfigMixin`. 
	`FlaxSchedulerMixin` inherits from `PushToHubMixin`

The inheritance results in the scheduler being able to upload to Huggingface (`PushToHubMixin`).
The ability to save and load scheduler settings (`FlaxSchedulerMixin`).
Configuration storage, save and load configs (`ConfigMixin`).

Two other classes are needed:
- `DPMSolverMultistepSchedulerState` the class that defines the state
- `FlaxDPMSolverMultistepSchedulerOutput` The class the defines the output dict in place of a tuple by holding the state. As such it is not important.
	- Inherits from `FlaxSchedulerOutput`. This class just holds a jnp.array
#### Functions:

The main function is `step()`:
	It predict the sample at the previous timestep by DPM-Solver based on learned model outputs.

`step()` calls
- `dpm_solver_first_order_update()`
- `multistep_dpm_solver_second_order_update()`
- `multistep_dpm_solver_third_order_update()`

Another important function is
`convert_model_output()` 
- Converts the given model output to corresponding type that the algorithm (DPM-Solver / DPM-Solver++) needs
This is a very specific function to this scheduler

`add_noise()`
- Adds noise according to `add_common_noise()`

`create_state()`
- Creates the state class

`set_timesteps()`
- Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

`scale_model_input()`
- Ensures interchangeability with schedulers that need to scale the denoising model input depending on the current timestep

### Use in practise 

## Karras VE flax

`KarrasVeScheduler` is a stochastic sampler tailored o variance-expanding (VE) models. It is based on the [Elucidating the Design Space of Diffusion-Based Generative Models](https://huggingface.co/papers/2206.00364) and [Score-based generative modeling through stochastic differential equations](https://huggingface.co/papers/2011.13456) papers.

### Achievements

- Simpler design (they argue)
- FID 1.79 conditioned and 1.97 unconditioned.
- 35 evals from noise to img

### How it flows together

#### Classes:

The main class `FlaxKarrasVeScheduler` inherits from `FlaxSchedulerMixin` and `ConfigMixin`. 
	`FlaxSchedulerMixin` inherits from `PushToHubMixin`

The inheritance results in the scheduler being able to upload to Huggingface (`PushToHubMixin`).
The ability to save and load scheduler settings (`FlaxSchedulerMixin`).
Configuration storage, save and load configs (`ConfigMixin`).

Two other classes are needed:
- `KarrasVeSchedulerState` the class that defines the state
- `FlaxKarrasVeOutput` The class the defines the output dict in place of a tuple by holding the state. As such it is not important.
	- Inherits from `BaseOutput`. Holds something akin to an array but defined by selves
#### Functions:

The two main functions are: `step()` and `step_correct()`
- These two functions share input and output.

`step()` predicts the next time step `t-1` using the reverse SDE.
- This happens through 3 computations that does not require any other functions.

`step_correct()` corrects the prediction through multiple steps (can be seen below the code example).
- Basically the same just slightly altered 1 equation

`create_state()`
- Creates state using the class `KarrasVeSchedulerState`

`add_noise_to_input()`
- Explicit Langevin-like "churn" step of adding noise to the sample according to a factor $\gamma_i ≥ 0$ to reach a higher noise level $\sigma_{hat} = \sigma_i + \gamma_i*\sigma_i$.

`set_timesteps()`
- Sets the continuous timesteps used for the diffusion chain. Supporting function to be run before inference.


## Commonalities 

### Classes

They basically follow  the same class structure, so that is nice

### Functions

They all share the following functions:
- `step()`
	- It seems like many of the computations are given as input Karras VE compared to SDE VE.
- `create_state()`
	- Very similar in style. They all create the state using the state class using the create function. 
	- SDE VE and SPM-solver do some specific stuff as well.
- `set_timesteps()`
	- Updates the timesteps in state using the place function in state. The state is returned.

Karras VE and SDE VE share:
- `step_correct()`
	- Does mainly the same thing as `step()`
This most likely occurs in one of the multistep functions in DPM-solver



#### Noise adding

Karras VE adds noise through a function called `add_noise_to_input()`

SDE VE sets Sigma's that control Drift and Diffusion terms. But I'm not certain get where the noise gets added to samples during training as it doesn't have a dedicated function for this

DPM-solver uses `add_noise()` to add noise to the sample