
# How it flows together

## Classes:

The main class `FlaxScoreSdeVeScheduler` inherits from `FlaxSchedulerMixin`. 
	`FlaxSchedulerMixin` inherits from `PushToHubMixin`

The inheritance results in the scheduler being able to upload to Huggingface (`PushToHubMixin`),
and to save and load scheduler settings (`FlaxSchedulerMixin`)

Two other classes are needed:
- `ScoreSdeVeSchedulerState` the class that defines the state
- `FlaxSdeVeOutput` The class the defines the output dict in place of a tuple. As such it is not important.


## Functions:

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

## Math
The main math in `step_pred` is:
$$diffusion = \sqrt{sigma_t^2 - sigma_{t-1}^2}$$
$$drift = -diffusion^2 \cdot \text{model\_output}$$
$$\mu_{\text{prev sample}} = sample - drift$$
$$\text{prev\_sample} = \mu_{\text{prev sample}} + diffusion \cdot noise$$

Where 
- the *sample* is the "image" we are currently working with
- *noise* is just normal
- *model_out* is the output of the model, aka the score ((?))

### `step_correct` and `step_pred` use
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



# Main Class
## `ScoreSdeVeScheduler` or `FlaxScoreSdeVeScheduler`


### Class functions:
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

#### Jax only:
- `has_state`
	- Returns True
- `create_state`
	- Description: "Creates state using `ScoreSdeVeSchedulerState.create()` and passing it through `set_sigmas` (see `ScoreSdeVeSchedulerState` class for more info) "
	- Input: None
	- Returns: `ScoreSdeVeSchedulerState`

#### Torch only
- `scale_model_input`
- `add_noise`


# Supporting classes
## `ScoreSdeVeSchedulerState` (state)
Class that stores 3 variables:
1. `timesteps`: `Optional[jnp.ndarray]` = None
2. `discrete_sigmas`: `Optional[jnp.ndarray] `= None
3. `sigmas`: `Optional[jnp.ndarray]` = None

## `FlaxSdeVeOutput`
Class that stores 3 variables:
1. `state`: `ScoreSdeVeSchedulerState`
2. `prev_sample`: `jnp.ndarray`
3. `prev_sample_mean`: `Optional[jnp.ndarray]` = None



# Inheritance classes

## `PushToHubMixin`

This is the very deepest inheritance class.
It has the functions:
- `_upload_folder`
	Uploads all files in `working_dir` to `repo_id`.
- `push_to_hub`
	Upload model, scheduler, or pipeline files to the 🤗 Hugging Face Hub

The class can be found at: `src\diffusers\utils\hub_utils.py` (identical for Flax and torch)

Neither of these need be changed. Therefore we quickly skip over it.

The important class that inherits from this class is:
## `SchedulerMixin` or `FlaxSchedulerMixin`

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



