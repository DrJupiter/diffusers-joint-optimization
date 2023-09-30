

# Abstract

We argue that the theory and practice of diffusion-based generative models are currently unnecessarily convoluted and seek to remedy the situation by presenting a design space that clearly separates the concrete design choices. 
This lets us identify several changes to both the sampling and training processes, as well as preconditioning of the score networks. 
Together, our improvements yield new state-of-the-art FID of 1.79 for CIFAR-10 in a class-conditional setting and 1.97 in an unconditional setting, with much faster sampling (35 network evaluations per image) than prior designs. 
To further demonstrate their modular nature, we show that our design changes dramatically improve both the efficiency and quality obtainable with pre-trained score networks from previous work, including improving the FID of a previously trained ImageNet-64 model from 2.07 to near-SOTA 1.55, and after re-training with our proposed improvements to a new SOTA of 1.36.

# Contributions

1. The goal is to obtain better insights into how these components are linked together and what degrees of freedom are available in the design of the overall system.
	- We focus on the broad class of models where a neural network is used to model the score of a noise level dependent marginal distribution of the training data corrupted by Gaussian noise. Thus, our work is in the context of denoising score matching

2. They :
	1. Identify the best-performing time discretization for sampling, 
	2. apply a higherorder Runge–Kutta method for the sampling process, 
	3. evaluate different sampler schedules, and 
	4. analyze the usefulness of stochasticity in the sampling process.
	-  This results in a significant drop in eval steps for good img. 
	- It can be used on pre-trained models

3. Architecture.
	- Use DDPM and NCSN
	- They provide principled analysis of the preconditioning of the networks'
		1. inputs
		2. outputs and
		3. loss
	- They use this to derive best practices for improving the training dynamics
	- They suggest a different distribution of noise levels during training
		- They also note that non-leaking augmentation works in diffusers as well

# Common framework

## Definitions

Data distribution: $p_{data}(x) \sim N(0,\sigma_{data})$

Mollified distributions: $p(x;\sigma)$
- Achieved by adding noise $N(0,\sigma)$ to the data $x$
- When $\sigma_{max} >> \sigma_{data}$ the $p(x;\sigma)$ is indistinguishable from $N(0,\sigma)$

The idea in diffusion is:
1. to sample from $N(0,\sigma_{max}^2 I)$
2. then sequentially de-noise it step by step: $\sigma_0 = \sigma_{max} > \sigma_1 > \dots > \sigma_N = 0$.
	- Such that we end at $x$ without noise

## ODE formulation

Yang Song (SDE diffusion) present a probability flow ODE.
This ODE adds or removes noise as we got forward or backwards in time.

To define the ODE we first choose: a schedule: $\sigma(t)$ 
- That defines the desired noise level at time $t$.
- The choice of scheduler is very important, and need not rely on nature.

### Evolving a sample: 
- $x_a \sim p(x_a; \sigma(t_a))$ from time $t_a$ to time $t_b$ yields $x_b \sim p(x_b; \sigma(t_b))$
	- The direction of evolution matters not
- Yang Song says that this requirement is satisfied by ODE:
$$dx = -\dot \sigma(t)~ \sigma(t) ~\nabla_x \log p(x;\sigma(t)) ~~dt$$
- Where the "dot" denotes time derivative,
	- so $\dot \sigma(t) = \cfrac{d}{dt}\sigma(t)$
- $\nabla_x \log p(x;\sigma(t))$ is the score function
	- a vector field that points towards higher density of data at a given noise level'
- Intuitively, an infinitesimal forward step of this ODE nudges the sample away from the dense data regions (meaningful data), at a rate that depends on the change in noise level
	- Backwards is the reverse

### Denoising score matching

The score function does not depend on the normalization constant of the underlying density function  $p(x;\sigma(t))$. 
- $p(x;\sigma(t)$ is generally intractable, which is why this is important
- Making the score function it easier to evaluate

Assuming:
- denoiser : $D(x;\sigma)$
	- That minimizes the $L_2$ error for samples drawn from $p_{data}$ separately for every $\sigma$:
$$E_{y \sim p_{data}} E_{n \sim N(0,\sigma^2 I)} ~~||D(y+n;\sigma)-y||^2_2$$
- Then the score is:
$$\nabla_x \log p(x;\sigma(t)) = \cfrac{D(x;\sigma)-x}{\sigma^2}$$
- Where:
	- $y$ is the training image (perturbed)
	- $n$ is the noise (Gaussian)

Here the score function can be interpreted as isolating the noise component from the signal in $x$. The SDE equation can then amplifies (or diminishes) the noise component over time.

The denoiser can be implemented as a NN $D_{\theta} (x;\sigma)$ to predict this value
- Which is what we do

### Time dependent signal scaling