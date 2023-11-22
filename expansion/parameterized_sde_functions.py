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
    t1 = Sigma.subs(t,1) != sp.eye(Sigma.shape[0])
    
    if not t0:
        print(f"Sigma(0)={Sigma.subs(t,0)} not {sp.zeros(Sigma.shape[0])}")
    if not t1:
        print(f"Sigma(1))={Sigma.subs(t,1)} Which should'nt be {sp.zeros(Sigma.shape[0])}")

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

        # print(eigval)

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
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def from_01_to_minfinf(x):
    return -sp.ln(1/x-1)

def minfinf_to_0minf(x):
    return -sp.exp(x)

def polynomial(x,params):
    summ = 0
    for i,param in enumerate(params):
        summ += param*x**i
    return summ

def exp_like_func(x):
    return sp.exp(x)/(sp.exp(x) + 1)

t1 = from_01_to_minfinf(t)


a,b,c,d,t,x,z = sp.symbols("a,b,c,d,t,x,z",real=True)
# params = [a,b,c,d] # 3rd degree poly
params = [1,2,3,1]

# Taylor of func
def taylor_of_f(f,step,eval_point,x):
    f_mi = sp.diff(f,x,step)
    diff = (x-eval_point)**step
    return f_mi/sp.factorial(step)*diff

# params = [taylor_of_f(exp_like_func(t), n, t, t).simplify() for n in range(5)]
# params = [taylor_of_f(t**2, n, t, t).simplify() for n in range(5)]
# params = [1/sp.factorial(n) for n in range(200)]

t2 = minfinf_to_0minf(polynomial(t1,params))
# t2 = minfinf_to_0minf(exp_like_func(t1).simplify())

t2

#%%
polynomial(t,params)

#%%
import numpy as np

N = 100
eps = 1e-6
linspace = np.array(range(N),dtype=np.float32)/(N)+eps # 100 values in [0,1]

y_space = np.array([-t2.subs(t,t_i).evalf() for t_i in linspace])
# y_space = np.array([(sp.exp(1)*t**2).subs(t,t_i).evalf() for t_i in linspace])


plt.plot(linspace,y_space)
plt.yscale("log")
plt.ylabel("negative value (for log scale)")
plt.xlabel("t in ]0,1[")
plt.show()

# %%


min(y_space),max(y_space)

y_space[np.argmin(np.abs(y_space-1))] # cloest to 1







#%%
t1 = from_01_to_minfinf(t)

a0,a1,a2,b0,b1,b2,c0,c1,c2,d0,d1,d2,t = sp.symbols("a0,a1,a2,b0,b1,b2,c0,c1,c2,d0,d1,d2,t",real=True)
params0 = [a0,b0,c0] # 2rd degree poly
params1 = [a1,b1,c1] # 2rd degree poly
params2 = [a2,b2,c2] # 2rd degree poly

f0 = minfinf_to_0minf(polynomial(t1,params0))
f1 = minfinf_to_0minf(polynomial(t1,params1))
f2 = minfinf_to_0minf(polynomial(t1,params2))

FF = sp.Matrix(
    [
        [f0,0,0],
        [0,f1,0],
        [0,0,f2]
    ]
)

test1_commuting_Ft(FF)
# %%

u = sp.symbols("".join([f"u_{i} " for i in range(9)]),real=True)

A_f = sp.Matrix(
    [
        [u[0],u[1],u[2]],
        [u[3],u[4],u[5]],
        [u[6],u[7],u[8]]
    ]
    )

A_d = sp.Matrix(
    [
        [u[0],0,0],
        [0,u[1],0],
        [0,0,u[2]]
    ]
)

test1_commuting_Ft(A_d@FF@A_d.T)

# %%
A_d@FF@A_d.T
# %%

test3_covar_pos_definite(A_d.subs({u[0]:1,u[1]:2,u[2]:3}))
# %%
lf0 = polynomial(t,params0)
lf1 = polynomial(t,params1)
lf2 = polynomial(t,params2)

val = 1
d = {a0:val,a1:val,a2:val,b0:val,b1:val,b2:val,c0:val,c1:val,c2:val}

L = sp.Matrix(
    [
        [lf0,0,0],
        [0,lf1,0],
        [0,0,lf2]
    ]
)

SS = sp.integrate(L@sp.eye(3)@L.T,t)



#%%
test2_1_covar_time(SS)
test3_covar_pos_definite(SS.subs(d))
list(SS.eigenvals().keys())
#%%

# %%


Lt = sp.Matrix(
    [
        [t**2,sp.sin(t),sp.cos(t)],
        [t**3,1/t,t/(t+1)],
        [a0,t,t**(-1)]
    ]
)

LL = Lt@np.eye(3)@Lt.T

print("LL sym = ",LL==LL.T)

LL

# %%

sp.integrate(Lt,t)
# %%


Lt = sp.Matrix(
    [
        [t,t**2,t**3],
        [t**4,t**5,t**6],
        [t**7,t**8,t**9],
    ]
    )

At = sp.Matrix(
    [
        [0,0,0],
        [0,a1,0],
        [0,0,0]
    ]
)

At@Lt@Lt.T@At.T
# %%

Lt = sp.Matrix(
    [
        [t,0,-t],
        [0,-t,0],
        [-t,0,-1],
    ]
    )

Lt@Lt.T
# %%
