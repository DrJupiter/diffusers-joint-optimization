#%%
import sympy as sp
import numpy as np

a,b,c,x,t = sp.symbols("a,b,c,x,t", real=True)

#%%
### Define testing functions ###

def test1_commuting_Ft(Ft):
    if Ft@sp.integrate(Ft,(t,0,t)) == sp.integrate(Ft,(t,0,t))@Ft:
        print("F(t) commutes with its integral")
    else:
        print("F(t) does NOT commute with its integral")


def test2_1_covar_time(Sigma):
    t0 = Sigma.subs(t,0) == sp.zeros(Sigma.shape[0])
    t1 = Sigma.subs(t,1) == sp.eye(Sigma.shape[0])
    
    if not t0:
        print(f"Sigma(0)={Sigma.subs(t,0)} not {sp.zeros(Sigma.shape[0])}")
    if not t1:
        print(f"Sigma(1))={Sigma.subs(t,1)} not {sp.eye(Sigma.shape[0])}")

    if t1 and t0:
        print("Sigma upholds the correct values at t=0 and t=1")


def test2_2_mean_time(mu):
    t0 = mu.limit(t,0) == sp.ones(mu.shape[0])[:,0]
    t1 = mu.limit(t,1) == sp.zeros(mu.shape[0])[:,0]
    
    if not t0:
        print(f"mu(0)={mu.limit(t,0)} not {sp.ones(mu.shape[0])[:,0]}")
    if not t1:
        print(f"mu(1))={mu.limit(t,1)} not {sp.zeros(mu.shape[0])[:,0]}")

    if t1 and t0:
        print("mu upholds the correct values at the t=0 and t=1")


def test3_covar_pos_definite(Sigma):

    # symmetry test
    if not Sigma == Sigma.T:
        print("Sigma is NOT symmetric which is required for it to be positive definite")
        return None

    # eigenval test
    eigenvals = list(Sigma.eigenvals().keys())
    for eigval in eigenvals:

        print(eigval)

        for i in np.linspace(0,1-1e-6,100): # Very hacky to make the range [0,1[ instead of [0,1]
            if not eigval.subs(t,i) >= 0:
                print(f"Sigma is not positive definite, it has at least one negative eigenval when evaluated with t \in [0;1]. Exactly at t={i} was the first instance encountered with value = {eigval.subs(t,i)}")
                return None

    print("Sigma is positive definite")


N = 3

# Dense matrix
# f_names = [[f"f_{i}{j} " for j in range(size)] for i in range(size)]
# f_symbols = [sp.symbols("".join(f_names[i])) for i in range(size)]
# Ft = sp.Matrix(f_symbols)

# F Diagonal matrix
# f_names = [f"f_{i} " for i in range(N)]
# f_symbols = sp.symbols("".join(f_names))

f = -1/(t**(-1)-1)
print(f"{f=}")

inner_f = sp.diff(f,t) #sp.diff(-1/(t-1),t)
print(f"{inner_f=}")

Ft = sp.Matrix([
    [inner_f,0,0],
    [0,inner_f,0],
    [0,0,inner_f]
    ], real=True)
print(f"{Ft=}")


# L Diagonal matrix
# l_names = [f"l_{i} " for i in range(N)]
# f1,f2,f3 = sp.symbols("".join(l_names))
Lt = sp.Matrix([
    [inner_f,0,0],
    [0,inner_f,0],
    [0,0,inner_f]
    ], real=True)
print(f"{Lt=}")


# define x0
x0 = sp.ones(N)[:,0] # R^(N x 1)

# calculate mu
mu = sp.integrate(Ft,(t,0,t)).exp()@x0 # R^(N x 1)
print(f"{mu=}")

# define Q as I
Q = sp.eye(N)

# calculate Sigma
Sigma = sp.integrate(Lt@Q@Lt.T,(t,0,t)) # R^(N x N)
print(f"{Sigma=}")


# Calculate A for checks 
# TODO: do this differently it runs very slowly
# a_names = [[f"a_{i}{j} " for j in range(size)] for i in range(size)]
# a_symbols = [sp.symbols("".join(a_names[i])) for i in range(size)]
# A = sp.Matrix(a_symbols)
# sp.solve(A@A.T-Sigma)

test1_commuting_Ft(Ft)
test2_1_covar_time(Sigma)
test2_2_mean_time(mu)
test3_covar_pos_definite(Sigma) # make it also take a list of variables and then the function should test if it overholds the things for all values of those variables.

# %%
import sympy as sp
t = sp.symbols("t", real=True)
k = 2
f = (1)/(t**k-1)
sp.limit(f,t,1,"-")

#%%

inner_f = sp.diff(f,t)

intf = sp.integrate(inner_f,t)

sp.limit(intf,t,1), sp.limit(f,t,1)

# %%

for i in np.linspace(-2,2,100):
    if 1**i != 1:
        print(1**i,i)


# %%
(-1/(t**(-3)-1)).simplify()
# %%
