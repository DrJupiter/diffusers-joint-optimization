

# Restrictions

## Restriction 1:   Commuting
The function $F(t)$ must commute with its integral $\int_{0}^{t}F(\tau)d\tau$ over the time interval from $0$ to $t$
$$F(t)\int_{0}^{t} F(\tau)d\tau = \int_{0}^{t}F(\tau)d\tau F(t)$$
This restriction results in a general solution for the mean and covariance

### Commuting rules:
Two matrices commute if $AB = BA$

If the product of two [symmetric matrices](https://en.wikipedia.org/wiki/Symmetric_matrix "Symmetric matrix") is symmetric, then they must commute. 
That also means that every diagonal matrix commutes with all other diagonal matrices.[8](https://en.wikipedia.org/wiki/Commuting_matrices#cite_note-8)[9](https://en.wikipedia.org/wiki/Commuting_matrices#cite_note-9)

### Commuting with your own integral rules

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

## impact

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
$\mu(t=1)=0$

At the same time we have that 
$\pmb{\exp}\left(\int_{0}^{t} F(\tau)d\tau\right)=0 ~~|~~t=1$
meaning that 
$\int_{0}^{t} F(\tau)d\tau=-\infty ~~|~~ t=1$

$\hat F(t)-\hat F(0)=-\infty ~~|~~ t=1$

This means that one of the two (or both) need to have a limit that as they go towards their thing they need to become $-\infty$

There could be some problems with finding $\hat F(0) = \inf$
So we look for 
$\hat F(t) = -\inf ~~|~~t=1$

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