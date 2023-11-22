

# Restrictions

## Restriction 1:   Commuting
The function $F(t)$ must commute with its integral $\int_{0}^{t}F(\tau)d\tau$ over the time interval from $0$ to $t$
$$F(t)\int_{0}^{t} F(\tau)d\tau = \int_{0}^{t}F(\tau)d\tau F(t)$$
This restriction results in a general solution for the mean and covariance

### Commuting rules:
Two matrices commute if $AB = BA$

If the product of two [symmetric matrices](https://en.wikipedia.org/wiki/Symmetric_matrix "Symmetric matrix") is symmetric, then they must commute. 
That also means that every diagonal matrix commutes with all other diagonal matrices.[8](https://en.wikipedia.org/wiki/Commuting_matrices#cite_note-8)[9](https://en.wikipedia.org/wiki/Commuting_matrices#cite_note-9)

### Diagonal tricks



## Restriction 2:   t=0: Covar and mean
Ensuring the multivariate distribution is a single point at the initial time

We let the initial point be denoted $x(0)$.
Thus the mean and the covariance are the following:
- $\pmb{\mu}(0) = x(0)$
- $\pmb{\Sigma}(0) = 0$
We now guarantee at the initial timestep the distribution is a single point.

## Restriction 2.5:   t=1: Covar and mean
Ensuring the multivariate distribution is a standard normal distribution at the final time.
- $\pmb{\mu}(1) = 0$
- $\pmb{\Sigma}(1) \neq 0$ 

## Restriction 3: Covar -> positive definite
The covariance must be positive definite at all times greater than the initial time

Klaus' text also says that maybe it only has to be semi-definite.
positive definite is a stronger term than positive semi-definite.

### Positive definite 
A square matrix [is called positive definite] if it is [symmetric] and all its [eigenvalues λ are positive], that is λ > 0

or test if
$vAv\geq 0~~|~~ \forall v$

### Symmetric
A matrix is [symmetric] if $A=A^T$
It can also be stated that a symmetric matrix is mirrored in the diagonal.

## Restriction 4: Q must be symmetric
Q must be symmetric
It is also nice if it is positive definite
Easiest to choose it as identity
- $Q=I$
# Formulas

$\pmb{\mu}(t) = \pmb{\exp}\left(\int_{0}^{t} F(\tau)d\tau\right)x(0)$
$\pmb{\Sigma}(t) = \int_{0}^{t} L(s)Q L(s)^{T}ds$

# Deriving formulas for parameterised Drift

Since we want
$\mu(t=0)=x_0$

We need to choose F such that
$\pmb{\exp}\left(\int_{0}^{t} F(\tau)d\tau\right)=1 ~~|~~t=0$
The solution will be of shape $e^{something}$
$\int_{0}^{t} F(\tau)d\tau= 1~~|~~t=0$

we define $\hat F(t) = \int F(\tau) d\tau$

As such we can write
$\hat F(0)-\hat F(0)=0$
This is always the case 


We also want 
$$\mu(t)\rightarrow 0  ~~|~~ t\rightarrow 1$$

At the same time we have that 
$$\pmb{\exp}\left(\int_{0}^{t} F(\tau)d\tau\right) \rightarrow 0 ~~|~~t \rightarrow 1$$
meaning that 
$$\int_{0}^{t} F(\tau)d\tau \rightarrow -\infty ~~|~~ t \rightarrow 1$$

$\hat F(t)-\hat F(0) \rightarrow -\infty ~~|~~ t \rightarrow 1$

This means that one of the two (or both) need to have a limit that as they go towards their thing they need to become $-\infty$

There could be some problems with finding $\hat F(0) = \inf$
So we look for 
$$\hat F(t)  \rightarrow  -\inf ~~|~~t \rightarrow 1$$

This could be:
$\hat F(t) = \cfrac{-1}{t-1}$
- But this would actually not be the case as our interval is \[0;1\], Tus we must approach from the left (-)
- As such the correct one would be
$\hat F(t) = \cfrac{1}{t-1}$

To find the corresponding $F(t)$ we differentiate $\hat F(t)$
and get:
$F(t) = \cfrac{1}{(1-t)^2}$

Following the same idea we can define
$\hat F(t) = \cfrac{1}{t^k-1}$

as $t^k=1 ~|~\forall k \in R / {0}$ 
- the k=0 wont work as we get division by 0. It is not the limit towards 1.

There is one problem here. The direction we take the limit from is important to if we get -inf or inf. 
Since our interval is \[0;1\], we approach from the left (1) in our limit as this is the only side that is within our interval.
This will result in $\hat F(t) = -\infty$ for $k<0$.
Therefore all is good

## Bijective map from $[0,1[ \rightarrow [0,\infty[$

Applying parameters in the space $[0,1]$ would result in constrained optimization, which is hard in NNs.
Therefore we wish to map 

Functions that satisfy this map are:
$\tan(x*pi/2)$
$\cfrac{1}{x-1}$
$\cfrac{x}{x-1}$
$\cfrac{x}{\sqrt{x^2-1}}$

## Bijective map from $]0,1[ ~~\rightarrow ~~]-\infty,\infty[ ~~\rightarrow ~~]0,-\infty[$

This would allow true unconstrainted use of parameters in the middle space

## Bijective map from $]0,1[ ~~\rightarrow ~~]-\infty,\infty[$

A function that does this is:
$f(x)=1-\ln(1/x-1)$

We need to apply parameters such that the result will go to $+\infty$
else we cannot guarantee that $t=1$ maps to $-\infty$

If we want to use a polynomial, we must ensure that it follows this property. 
This could be done by letting the quickest scaling factor have forced positive coefficient.
This would result in constraint optimization if we don't do something to it.
One way would be to take the absolute value of it. This however still allows for it to become 0, and that would break the math.
Another way is to not have it as a parameter. 
This would result in a slightly less flexible model, but would maintain the math without fault.

An example could be

We define:
$t_1 = 1-\ln(1/x-1)$
$t_1 \in ]-\infty;\infty[$

With parameters
$a,b,c,d \in R$

Then the polynomial
$f(x) = a + b t_1 + c t_1^2 + d t_1^3 +t_1^4$


### Bijective map from $]-\infty,\infty[ ~~\rightarrow ~~]0,-\infty[$

A function that does this is:
$g(x)=-e^x$

### Combined $]0,1[ ~~\rightarrow  ~~]0,-\infty[$

$g(f(x)) = -e^{1-\ln(1/x-1)}$

$-e^{1-\ln(1/x-1)}=\cfrac{-e^1}{e^{\ln(1/x-1)}}=\cfrac{-e}{1/x-1}$
$e^{\ln(1/x-1)}=1/x-1$


$\cfrac{-e}{1/x-1} = \cfrac{-ex}{x-1}$


### Other possible functions:


#### 1:
$\cfrac{-x}{x-1} = \cfrac{-1}{1/x-1}$
		  $= \cfrac{-1}{e^{\ln(1/x-1)}}$
		  $= -e^{-\ln(1/x-1)}$

This results in the map
$]0,1[ ~~\rightarrow ~~]-\infty,\infty[$
being
$$-\ln(1/x-1)$$

and the map from
$]-\infty,\infty[ ~~\rightarrow ~~]0,-\infty[$
being
$$-e^x$$


#### 2:
$\cfrac{-1}{x-1}$
$= \cfrac{-1/x}{1/x-1}$
$= \cfrac{-e^{\ln(-1/x)}}{e^{\ln(1/x-1)}}$
$= -e^{ln(1/x)-\ln(1/x-1)}$
$= -e^{ln(\cfrac{1/x}{1/x-1})}$
$= -e^{ln(\cfrac{1}{1-x})}$

This results in the map
$]0,1[ ~~\rightarrow ~~]-\infty,\infty[$
being
$$\ln(1/(1-x))$$

and the map from
$]-\infty,\infty[ ~~\rightarrow ~~]0,-\infty[$
being
$$-e^x$$

#### 3:

$-\ln(1/(1-x))=z$

$1/(1-x)=e^z$
$1=e^z*(1-x)$
$1=e^z- x*e^z$
$\cfrac{-1+e^z}{e^z}=x$
$-1/e^z+1=x$

## Bijective mapping of $[0,\infty[\rightarrow[0,1[$

The amount of functions that easily fulfill the above are limited.
It would be nice if we instead could use functions like polynomials.

Therefore we look into using a bijective mapping from $R_+$ to $[0,1[$.
This projects any function we could desire into our space, excluding $1$ but that is a willing sacrifice.

Such a function could be:

$$1-\cfrac{a}{f(x)+a} ~~~~ (1)$$
Which maps to $[0,1[$
or
$$\tan^{-1}(a*f(x))~~~~ (2)$$
Which maps to $[0,1.571[$
- $1.571$ is the maximum value achieved when the maximum value of float64 is used:
	- 1.571 = arctan(1.7E+308)
- This can be transformed to the correct space by scaling with factor
- $1/1.571$
$$\cfrac{\tan^{-1}(a*f(x))}{1.571}~~~~ (3)$$
Which maps from $[0,1[$

These both hold to this mapping.




# Deriving formulas for parameterised diffusion

We want to figure out how to choose $L(s)$ such that the restrictions 2, 2.5, 3, met
assuming $Q=I$ (such that 4 is already met).

Since most constraints fall on the covariance and not $L(s)$ we work back from that:

The covariance is given by:
$\pmb{\Sigma}(t) = \int_{0}^{t} L(s)Q L(s)^{T}ds$

We require that 
- $\Sigma(t)$ is positive definite 
and that:
- $\pmb{\Sigma}(0) = 0$
- $\pmb{\Sigma}(1) \neq 0$ 

Lets first look at the time-value constraints.

## Constraint $\pmb{\Sigma}(0) = 0$
Similar to the drift, we have that when we integrate from 0 to 0, we always have 0:
So this constraint will be fulfilled not matter what shape $L(s)$ takes.

## Constraint $\pmb{\Sigma}(1) \neq 0$

This constraint can be fulfilled in multiple different ways:

### Squared values and additive term

We could square all the functions in $\Sigma$, such that they will always be $\geq 0$. And then we could add a additive term to the functions.
This forces all values to be positive in the function as such the integral cannot be 0.

The reason we need to square the function first, is else we could run into a situation where the values are 0, or exactly equalize the additive term.

### Functions where $|f(t)|>0 ~~|~~0 < t \leq 1$

Per restriction $\Sigma(t) ~~|~~t>0$ must be positive definite

One way to do this is making sure $\Sigma(t)$ is fully positive.

Looking at $\Sigma$ we see that:

$\Sigma(t) = (\int L L^T)(t=t) - (\int L L^T)(t=0)$
- Saying that the integral is evaluated in t=

If we want $\Sigma$ to be fully positive we must have
- $(\int L L^T)(t=t) > (\int L L^T)(t=0)$

A way to ensure this is to [square every function (or maybe only parameters)] and say that [every function must obey]:
- $|f(t)|>0 ~~|~~0 < t \leq 1$
Or 
- $f(t)^2>0 ~~|~~ 0<t\leq 1$

It is important to consider that this must also be the case for every parameter.

$f(t,\theta)^2>0 ~~|~~ 0<t\leq 1~,~ \forall \theta \in R$

This means polynomials consisting of multiple parts are a problem as they can have stationary points in this interval.
- Unless we square the parameters individually, creating monotonically increasing functions.
- This would also result in the requirement being satisfied.


#### Diagonal $L$

In the case of a diagonal $L$ this is forced

If it is not, then squaring the values could be a good idea

This would also force the matrix to always be [positive definite]


We know that any totally positive matrix has only positive eigenvalues

and that $A A^T$ results in a symmetric matrix

So if we let all values in $L$ be positive, we will guarantee a positive definite matrix. Not matter if it is diagonal or not.

One way we could do this is simply by squaring all values



## Forcing positive definite 

It might be possible to force any matrix to be positive definite by multiplying it with other matrices

Or we can take all parameters squared in the math

Again we have that n

### Form

#### AL Form
We define a matrix $A$ that stores parameters

We then define $L = AL$
such that
$LL^T = ALL^TA^T$
When we int we can then
$A \int (L L^T) A^T$

This way we can apply parameters easily to everything.

Additionally we know that if $LL^T$ is positive definite, then $ALL^TA^T$ is as well

Using this method we would define each entry in $L$ as a singular function
The dot product with $A$ would then parameterize and distribute it.

#### Fully parameterized form

This way each entry is a sum of function with their own parameters like:
$A[0,0]=a*t^2+b*\sin(t)+c*e^t$
Each entry has its own set of parameters, controlling which functions are expressed how much.

This allows direct control of how it works







# Remember

## Differentiable
The functions in L(t) and F(t) must be differentiable and thus continuous in the interval t=\[0;1\]

## Inspect A(t)
Remember to inspect $A(t)$ and $A(t)^{-1}$ to make sure these are not poorly defined, as we need to evaluate them directly

## Parameterized functions of t
If functions take a parameter that modify the function where t is present, restrictions 2 and 2.5 should still be overheld.

sin(t), t=0 -> sin(0)=0 which could be valid for Covar at t=0
But if we have:
sin(t+a),t=0 -> sin(a) $\neq$ 0
which could be a problem

or
$\sin(a*t*\pi/2)$ at $t=1,a=0$ -> $\sin(0) = 0$
but if a=1,t=1
$\sin(1*1*\pi/2)=1$

### Sol 1
One way to solve this is not to modify that part of the function, but that reduces flexibility

### Sol 2
Another way is to just make sure it always equals out in the 2 edge cases.

## constrained optimization

When adding multiple functions together they should still sum to 1 in the mean case

The same might apply for Covar


## Forcing LQL to be positive definite

ALQLA where A is some matrix that is frozen.

## Make sure functions in F and L are differentiable with respect to parameters