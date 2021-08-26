
# %%

import numpy as np 
import scipy as sp 
import numpy.random as rand

# %% [markdown]

# ## This notebook is about Bayesian regression assuming Gaussian distribution

# %% [markdown]

# ### First define the regression problem and generate data
# ### Linear regression with parameters/weights $w$ 
# > ### $y_i = x_i^T w + \epsilon$
# > ### where $\epsilon$ is measurement noise $\sim \mathcal{N}(0, \sigma_n^2)$.

# %%


# p parameters/weights
p = 10
w = np.zeros(p)
rand.seed(0)  # set seed for parameters.
for i in range(p):
	w[i] = rand.random()  # random values b/w 0 and 1.

# generate data
N = 100  # no.of data points

# noise
rand.seed(2)  # set seed for noise.
v = rand.normal(loc=0.0, scale=1.0, size=(N,)) 

X = np.zeros([N,p])
for i in range(p):
	rand.seed(i+10)  # set seed for i-th data.
	X[:,i] = rand.normal(rand.randint(5,15), scale=2.0, size=N)

Yt = np.dot(X, w)  # compute outputs
Y = Yt + v  # noisy outputs

print("No.of parameters: p=%d..\n" %(p))
print("Length of data generated: N=%d..\n" %(N))
print("Parameter (w) list: %s..\n" %(str(w)))
print("Y is of size %d times 1..\n" %(N))
print("X is of size %d times %d..\n" %(N,p))

# %% [markdown]

# ### Use GPR to estimate parameters $w$ from data $(X,Y)$
# > ### $y_i = x_i^T w + \epsilon$
# > ### where $\epsilon$ is measurement noise $\sim \mathcal{N}(0, \sigma_n^2)$.
#
# ### In terms of the data matrices, 
# > ### $Y = X \cdot w + e$ 
# > ### Given $X$ and $w$, the probability distribution of $Y$ (likelihood function) can be obtained as 
# >> ### $p(Y|X,w) \sim \mathcal{N}(X \cdot w, \sigma_n^2 \cdot I)$.
#
# ### Also we can assume some prior distribution for the parameters $w$ which reflects available information on the parameters (in case no prior information is available, we may set them to be zero-mean with large variance),
# > ### $p(w) \sim \mathcal{N}(0,\Sigma_p)$
#
# ### We can use Bayes theorem to compute the posterior probability distribution given the prior and the likelihood,
# > ### $\text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{marginal likelihood}} $
# > ### or,
# > ### $p(w|Y,X) = \frac{p(Y|X,w) \cdot p(w)}{p(Y|X)}$
# > ### The denominator is independent of $w$ and acts like a normalization factor. 
# > ### Using the Bayes formula, the posterior can be computed to be proporational to 
# > ### $p(w|X,Y) \propto \exp \left\{ -\frac{1}{2}(w-\bar{w})^T \left( \frac{1}{\sigma_n^2} X^T X + \Sigma_p^{-1} \right) (w-\bar{w}) \right\} $, 
# > ### or,
# > ### $p(w|X,Y) \sim \mathcal{N}(\bar{w}, A^{-1})$,
# > ### where $A = \left( \frac{1}{\sigma_n^2} X^T X + \Sigma_p^{-1} \right)$, $\bar{w} = \frac{1}{\sigma_n^2} A^{-1} X^T Y$.
#
# ### Estimate of the weight vector can be taken as $\bar{w}$ and the variance (uncertainty) associated with it given by $A^{-1}$.
#
# ### Prediction (at point $x_*$): 
# > ### $p(f(x_*)|X,Y) \sim \mathcal{N}(x_*^T \bar{w}, x_*^T A^{-1} x_*)$
#
# ### Estimate of the value at a test point $x_*$ can be taken as $x_*^T \bar{w}$ and the variance (uncertainty) associated with it given by $x_*^T A^{-1} x_*$.

# %% [markdown]

# ### Get the posterior distribution of weights

# %%

# Define the parameters
sigma_n = 1.5  # output noise variance, original value = 1
Sigma_p = 10*np.eye(p)  # prior variance
Sigma_p_inv = np.linalg.inv(Sigma_p)

A = (1/sigma_n*sigma_n)*np.dot(X.transpose(), X) + Sigma_p_inv
A_inv = np.linalg.inv(A)

wbar = (1/sigma_n*sigma_n)*np.dot(A_inv, np.dot(X.transpose(), Y))

# also compute the least squares value for comparison
w_ls = np.linalg.inv(np.dot(X.transpose(), X))
w_ls = np.dot(w_ls, np.dot(X.transpose(),Y))

# %% [markdown]

# ### Predict the value at a test point

# %%

xs = np.zeros(p)
for i in range(p):
	xs[i] = rand.normal(rand.randint(5,15), scale=2.0)

fxs = np.dot(xs, wbar)
sigma_fxs = np.dot(xs, np.dot(A_inv, xs))

fxs_ls = np.dot(xs, w_ls)
fxs_true = np.dot(xs, w)

print("Test point (xs): %s..\n" %(str(xs)))
print("Value y=f(xs) obtained using GPR: (%.2f, %.2f)..\n" %(fxs, sigma_fxs))
print("Value y=f(xs) obtained using LS: %.2f..\n" %(fxs_ls))
print("Value y=f(xs) true: %.2f..\n" %(fxs_true))

# %% [markdown]

# ### This can be extended in a straight-forward way to the case the features are nonlinear,
# > ### $y_i = \phi(x_i)^T w + \epsilon$.
#
# ### In terms of data matrices,
# > ### $Y = \Phi(X)^T w + e$.
#
# ### In this case,
# > ### $p(f(x_*)|X,Y) \sim \mathcal{N}(\phi(x_*)^T \bar{w}, \phi(x_*)^T A^{-1} \phi(x_*))$
# > ### where $A = \left( \frac{1}{\sigma_n^2} \Phi(X)^T \Phi(X) + \Sigma_p^{-1} \right)$, $\bar{w} = \frac{1}{\sigma_n^2} A^{-1} \Phi(X)^T Y$.

# %%
