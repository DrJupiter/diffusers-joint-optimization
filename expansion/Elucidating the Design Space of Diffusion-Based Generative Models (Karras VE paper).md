

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
	2. apply a higher order Runge–Kutta method for the sampling process, 
	3. evaluate different sampler schedules, and 
	4. analyze the usefulness of stochasticity in the sampling process.
	- This results in a significant drop in eval steps for good img. 
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

Mollified distributions: $p(x;\sigma)$  (perturbed data)
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

### Evolving a sample (probability flow ODE): 
- Evolving a sample $x_a \sim p(x_a; \sigma(t_a))$ from time $t_a$ to time $t_b$ yields $x_b \sim p(x_b; \sigma(t_b))$
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
- $p(x;\sigma(t)$ is generally intractable, which is why this is important that we dont need to compute it
- Making the score function it easier to evaluate

Assuming:
- denoiser : $D(x;\sigma)$
	- That minimizes the $L_2$ error for samples drawn from $p_{data}$ separately for every $\sigma$:
$$E_{y \sim p_{data}} E_{n \sim N(0,\sigma^2 I)} ~~||D(y+n;\sigma)-y||^2_2$$
- Then the score is:
$$\nabla_x \log p(x;\sigma(t)) = \cfrac{D(x;\sigma)-x}{\sigma^2}$$
- Where:
	- $y$ is the training image (un-perturbed)
	- $n$ is the noise (Gaussian)

Here the score function can be interpreted as isolating the noise component from the signal in $x$. The SDE equation can then amplifies (or diminishes) the noise component over time.

The denoiser can be implemented as a NN $D_{\theta} (x;\sigma)$ to predict this value
- Which is what we do

### Time dependent signal scaling

Some methods introduce a new variable $s(t)$.
$s(t)$ is called a scaling variable as it is used to scale the noisy data: $x=s(t) \hat x$
- Where $\hat x$ is the noisy data.
Using this term changes the ODE as follows:
$$dx = \left[ \cfrac{\dot s(t) }{s(t)} x - s(t)^2\dot \sigma(t)~ \sigma(t) ~\nabla_x \log p\left(\cfrac{x}{s(t)};\sigma(t)\right) \right]~~dt$$
- Note that we explicitly undo the scaling of $x$ when evaluating the score function to keep the definition of $p(x; σ)$ independent of $s(t)$

## Solution by discretization

- Use definition of score and put it in the equation above
- Discretize with Runge Kutta
	- A second order method
	- Normally the first order method, Euler is used

## Putting it all together

Some of our formulas appear quite different from the original papers as [indirection] and [recursion] have been [removed]

![[different_SDEs.png]]

The main purpose of this reframing is to bring into light all the independent components that often appear tangled together in previous work

In our framework, there are no implicit dependencies between the components.
- Any choices (within reason) for the individual formulas will, in principle, lead to a functioning model.
- In other words, changing one component does not necessitate changes elsewhere in order to, e.g., maintain the property that the model converges to the data in the limit



# Deriving 

Yang Song define their [forward SDE] in [SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS](https://arxiv.org/abs/2011.13456) as:

$$d x(t) = f(x(t),t)dt + g(t) d \omega_t$$
- $x$ is the data with dims $R^d$
	- $x(t)$ is the data at perturbation lvl t
- $t$ is the continuous time
- $f$ is the drift coefficient of x(t). $f$ maps from $R^d \rightarrow R^d$
- $g$ is the diffusion coefficient of x(t). $g$ maps from $R^d \rightarrow R^{d \times d}$ or $R \rightarrow R$ in the scalar case.
	- We use scalar case here
- $\omega_t$ is the noise (The Wiener process AKA Brownian motion)

The SDE has a unique strong solution as long as the coefficients are globally Lipschitz in both state and time (Øksendal, 2003).

They say that 
$f(x,t)=f(t) x$
- $f(t) : R\rightarrow R$
is always the case, and as such they can input it in the function

$$d x(t) = f(t) x(t) dt + g(t) d \omega_t$$

They then look a the perturbation kernel for this SDE:
$$p_{0t}(x(t)~|~x(0)) = N(~x(t);~ s(t)x(0) ,~ s(t)^2 \sigma(t)^2 I~)$$
with:
$$s(t) = \exp\left(  \int_0^t f(\xi) d\xi   \right)$$
$$\sigma(t) = \sqrt{\left(  \int_0^t \cfrac{g(\xi)^2}{s(\xi)^2} d\xi   \right)}$$

## Relative to Yang Song:

Here we will use the general rules as above to get Yang Songs perturbation kernels

We relate to the general forms of
SDE:
$$d x(t) = f(t) x(t) dt + g(t) d \omega_t$$
Perturbation kernel
$$p_{0t}(x(t)~|~x(0)) = N(~x(t);~ s(t)x(0) ,~ s(t)^2 \sigma(t)^2 I~)$$

### VE SDE
The VE SDE is eq 30:
$$dx = \sigma_{min} \left(\cfrac{\sigma_{max}}{\sigma_{min}}\right)^t \sqrt{2 \log \left( \cfrac{\sigma_{max}}{\sigma_{min}} \right)} d\omega_t ~~|~~ t \in ~]0,1]$$
$$p_{0t}(x(t)~|~x(0)) = N(~x(t);~ x(0) ,~ [\sigma_y(t)^2 - \sigma_y(0)^2] I~)$$
- $\sigma_y$ for Yang Song, as to now confuse these sigmas with the other ones

We see in the SDE that
$f(t)=0$
which is also confirmed in the kernel
$$s(t) = \exp\left(  \int_0^t f(\xi) d\xi   \right) = 1 = \exp(0)$$
We see that 
$$g(t) = \sigma_{min} \left(\cfrac{\sigma_{max}}{\sigma_{min}}\right)^t \sqrt{2 \log \left( \cfrac{\sigma_{max}}{\sigma_{min}} \right)}$$
with
$$\sigma_d = \left(\cfrac{\sigma_{max}}{\sigma_{min}}\right)$$
$$g(t) = \sigma_{min} (\sigma_d)^t \sqrt{2 \log \sigma_d}$$

We input in our formula for sigma

$$\sigma(t) = \sqrt{\left(  \int_0^t \cfrac{g(\xi)^2}{s(\xi)^2} d\xi   \right)}$$
$$\sigma(t) = \sqrt{\left(  \int_0^t g(\xi)^2 d\xi   \right)}$$
$$\sigma(t) = \sqrt{\left(  \int_0^t \left( \sigma_{min} (\sigma_d)^{\xi} \sqrt{2 \log \sigma_d} \right)^2 d\xi   \right)}$$
$$\sigma(t) = \sqrt{\left(  \int_0^t \sigma_{min}^2 (\sigma_d)^{\xi^2} \sqrt{2 \log \sigma_d}^2 d\xi   \right)}$$
$$\sigma(t) = \sqrt{\left(  \int_0^t \sigma_{min}^2 (\sigma_d)^{2\xi} 2 \log \sigma_d d\xi   \right)}$$

$$\sigma(t) = \sqrt{\left(  \sigma_{min}^2 2 \log \sigma_d \int_0^t  (\sigma_d)^{2\xi} d\xi   \right)}$$
Using: $\int_0^t  (\sigma_d)^{2\xi} d\xi = \cfrac{\sigma_d^{2t}}{2 \log \sigma_d} - \cfrac{1}{2 \log \sigma_d}$
$$\sigma(t) = \sqrt{\left(  \sigma_{min}^2 2 \log \sigma_d \left(\cfrac{\sigma_d^{2t}}{2 \log \sigma_d} - \cfrac{1}{2 \log \sigma_d}\right)   \right)}$$
$$\sigma(t) = \sigma_{min} \sqrt{\left(\sigma_d^{2t} - 1\right)}$$


$$\sigma(t)^2 = \sigma_{min}^2 \left(\sigma_d^{2t} - 1\right)$$
$$\sigma(t)^2 = \sigma_{min}^2 \sigma_d^{2t} - \sigma_{min}^2$$
We know that from Yang Song
- $\sigma_d = \sigma_{min} \sigma_d^t$

$$\sigma(t)^2 = \left(\sigma_{min} \sigma_d^{t}\right)^2 - \sigma_{min}^2$$
Since we know that $x^0 = 1$
this is the same as Yang Songs  perturbation kernel.

### sub-VP SDE
SDE:
$$dx = -½ \beta(t) x dt + \sqrt{\beta(t)\left(  1-e^{-2 \int_0^t \beta(s) ds}  \right)}d\omega_t$$
with perturbation kernel:
$$p_{0t}(x(t)~|~x(0)) = N(~x(t);~ x(0) e^{-2 \int_0^t \beta(s) ds} ,~  [1-e^{-2 \int_0^t \beta(s) ds}]^2I~)$$

We read from the SDE that:

$$f(t)=-\cfrac{1}{2} \beta(t)$$
and
$$g(t) = \sqrt{\beta(t)\left(  1-e^{-2 \int_0^t \beta(s) ds}  \right)}$$

We first find s(t):
$$s(t) = \exp\left(  \int_0^t f(\xi) d\xi   \right)$$
Inserting f:
$$s(t) = \exp\left(  \int_0^t -\cfrac{1}{2} \beta(\xi) d\xi   \right)$$
$$s(t) = \exp\left(  -\cfrac{1}{2} \int_0^t \beta(\xi) d\xi   \right)$$
Which is what Yang Song has

Finding $\sigma(t)$
$$\sigma(t) = \sqrt{\left(  \int_0^t \cfrac{g(\xi)^2}{s(\xi)^2} d\xi   \right)}$$

Using:

$\alpha(t)=\int_0^{t} \beta(s) ds$
$\dot \alpha(t)=\beta(t)$

we have
$$s(t) = e^{\cfrac{-1}{2} \alpha(t)}$$
$$g(t) = \sqrt{\beta(t)\left(  1-e^{-2 \alpha(t)}  \right)}$$
Inserting we get
$$\sigma(t) = \sqrt{\left(  
\int_0^t \cfrac{\beta(\xi)\left(  1-e^{-2 \alpha(\xi)}  \right)}
{e^{-\alpha(\xi)}} d\xi
\right)}$$

Since we want to find
$s(t)^2 \sigma(t)^2$

We find the sigma part first

$$\sigma(t)^2 =  
\int_0^t \beta(\xi) \cfrac{\left(  1-e^{-2 \alpha(\xi)}  \right)}
{e^{-\alpha(\xi)}} d\xi$$

$$\sigma(t)^2 =  \int_0^t \beta(\xi) \left(  1-e^{-2 \alpha(\xi)}  \right) e^{\alpha(\xi)} d\xi$$

$$\sigma(t)^2 =  \int_0^t \beta(\xi) \left( e^{\alpha(\xi)} -e^{\alpha(\xi)} e^{-2 \alpha(\xi)}  \right) d\xi$$

$$\sigma(t)^2 =  \int_0^t \beta(\xi) \left( e^{\alpha(\xi)} -e^{-\alpha(\xi)}  \right) d\xi$$

$$\sigma(t)^2 =  \int_0^t \beta(\xi) e^{\alpha(\xi)} - \beta(\xi) e^{-\alpha(\xi)}  d\xi$$
Use difference rule for integration 

$$\sigma(t)^2 =  \int_0^t \beta(\xi) e^{\alpha(\xi)} d\xi - \int_0^{\xi}\beta(\xi) e^{-\alpha(\xi)}  d\xi$$

$$\sigma(t)^2 =  \int_0^t \beta(\xi) e^{\alpha(\xi)} d\xi + \int_0^{\xi} -\beta(\xi) e^{-\alpha(\xi)}  d\xi$$
using 
$\dot \alpha(t)=\beta(t)$

and inserting:

$\int_0^t \beta(\xi) e^{\alpha(\xi)} d\xi = e^{\alpha(t)}-e^{\alpha(0)} = e^{\alpha(t)} - 1$

Thus we get:

$$\sigma(t)^2 =  (e^{\alpha(t)} - 1) + (e^{-\alpha(t)} - 1)$$

$$\sigma(t)^2 =  e^{\alpha(t)} + e^{-\alpha(t)} - 2$$
remember we want to find
$s(t)^2 \sigma(t)^2$

$$s(t)^2 \sigma(t)^2 = e^{-\alpha(t)} (e^{\alpha(t)} + e^{-\alpha(t)} - 2)$$

$$s(t)^2 \sigma(t)^2 = 1 + e^{-2\alpha(t)} - 2 e^{- \alpha(t)})$$
Which is equivalent to if we insert alpha

$$[1-e^{-\int_0^t \beta(s) ds}]^2 = 1-e^{-\int_0^t \beta(s) ds}+e^{-2\int_0^t \beta(s) ds}$$

