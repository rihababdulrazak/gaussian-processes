
# %%

import numpy as np 
import scipy as sp 
import numpy.random as rand
import pandas as pd

# %%

#import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as go
plt.style.use('seaborn-whitegrid')  # some other styles - default, classic, bmh, fast, fivethirtyeight, ggplot, seaborn, seaborn-bright, seaborn-dark, seaborn-dark-palette, seaborn-darkgrid, seaborn-deep, seaborn-notebook, seaborn-ticks, seaborn-whitegrid, tableau-colorblind10
plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20

# %% [markdown]

# # This notebook is about Gaussian Process Regression.
# ## But first we will look at Gaussian random variables.
# ## A single Gaussian random variable is described by its mean value $m$ and variance value $\sigma^2$:
# > ## $p(y) = \frac{1}{\sqrt{2\pi \sigma}} \exp^{-\frac{(y-m)^2}{2 \sigma^2}} $.

# %%

y = np.linspace(0,10,1000)
p = (1/np.sqrt(2*np.pi))*np.exp(-0.5*(y-5)*(y-5))
t = go.Scatter(x=y, y=p, name='single Gaussian distribution')
fig1 = go.Figure(data=[t])
fig1.update_layout(title='Gaussian pdf', xaxis_title='y (value taken by the random variable)', yaxis_title='p(x) (probability density)')

# %% [markdown]

# ## The above plot shows a Gaussian distribution with $m=5$ and $\sigma=1$.

# %% [markdown]

# ## A multivariate Gaussian distribution is defined over more than one variables. 
# ## Say, there are two possibly (cor)related variables $y_1$ and $y_2$.
# ## Since they may be (cor)related, to evaluate the probability for $y_1$ and $y_2$, we need to consider both of them together (joint random vector ${\bf{y}} = [y_1, y_2]^T$) as a *joint distribution*:
# > ## $ p(y_1,y_2) = p({\bf y}) = \frac{1}{\sqrt{(2 \pi)^2 |\Sigma|}} \exp^{-\frac{1}{2} ({\bf y}-{\bf m})^T \Sigma^{-1} ({\bf y}-{\bf m})} $.
# ## In this case, the mean ${\bf m}$ is a vector ${\bf m} = [m_1, m_2]^T$ and $\Sigma$ is a $2 \times 2$ *covariance matrix*. If the variables are not (cor)related, the covariance matrix $\Sigma$ will be diagonal and the joint distribution collapses to two individual Gaussian distributions.

# %%

# create 2D grid
x = np.linspace(0,10,100)
y = np.linspace(0,10,100)
(Y1, Y2) = np.meshgrid(x,y)

# %%

# covariance matrix
S = np.array([[1,0], [0,2]])
Sinv = np.linalg.inv(S)
# mean vector
m = np.array([[4],[5]])
# pdf
p = np.zeros(Y1.shape)
for i in range(Y1.shape[0]):
	for j in range(Y1.shape[1]):
		Yi = np.array([[Y1[i,j]], [Y2[i,j]]])
		p[i,j] = (1/np.sqrt(4*np.pi*np.pi*2))*np.exp(-0.5*np.dot((Yi-m).transpose(), np.dot(Sinv,Yi-m)))

# %%

t1 = go.Scatter3d(
     x=Y1.flatten(), 
     y=Y2.flatten(), 
     z=p.flatten(),
     name='multivariate normal',
     mode='markers'
     )
fig2 = go.Figure(data=[t1])
fig2.update_layout(
     title = 'Multivariate Gaussian',
     width=800, 
     height=800,
     scene=dict(
         xaxis_title='y1',
         yaxis_title='y2',
         zaxis_title='p(y1,y2) [pdf]',
     ),
     margin=dict(l=0, r=0, b=0, t=0)  # tight layout
)

# %% [markdown]

# ## The above plot shows a joint Gaussian distribution with mean ${\bf m}=[4,5]^T$ and covariance matrix $\Sigma= \left( \begin{array}{cc} 1 & 0 \\ 0 & 2 \end{array} \right)$.
# ## Similarly, joint distribution can be defined for $k$ Gaussian variables (for any finite $k>0$):
# > ## $ p(y_1,\dots,y_k) = p({\bf y}) = \frac{1}{\sqrt{(2 \pi)^k |\Sigma|}} \exp^{-\frac{1}{2} ({\bf y}-{\bf m})^T \Sigma^{-1} ({\bf y}-{\bf m})} $.
# ## where $\bf{y}$ is the random vector ${\bf{y}} = [y_1, \dots, y_k]^T$.

# %% [markdown]

# # Now we come to a random process.
# ## A *random process* is a collection of random variables.
# ## A *Gaussian process* is a collection of random variables, where any finite number of them are *jointly Gaussian*. A typical example is a time-series, where corresponding to each time point $t$ we have a random variable $X_t$.
# ## In the context of Gaussian Process Regression, we wish to model the function 
# > ## $y=f(\bf{x})$       
# > ## ($\bf{x}$ is bold because it is considered a vector corresponding to multiple features)
# ## as a Gaussian process. i.e., given any finite set of points ${\bf x}_1,{\bf x}_2,\cdots,{\bf x}_k$, we model the joint distribution of ${\bf y} = [f({\bf x}_1),f({\bf x}_2),\cdots,f({\bf x}_k)]$ to be a Gaussian distribution with some mean, say $[m({\bf x}_1),m({\bf x}_2),\cdots,m({\bf x}_k)]$ and covariance matrix $[k({\bf x}_i, {\bf x}_j)]$.
# ## The Gaussian process $f(\bf{x})$ is completely specified by its *mean function* $m(\bf{x})$ and the *covariance function* $k(\bf{x},\bf{x}')$:
# > ## $f({\bf{x}}) \sim \mathcal{GP}(m({\bf{x}}), k(\bf{x}, \bf{x}'))$.
# > ## A common example of a covariance function is the squared exponential $k({\bf x}, {\bf x}') = e^{-\frac{\|{\bf x} - {\bf x}^{'}\|^2}{l^2}}$. The value of $l$ determines how fast the covariance between two points dies down as distance between them increases.

# %% [markdown]

# ## *The Gaussian process can be viewed as a probability distribution over functions*.
# > ## If we assume a Gaussian distribution, the we can sample points from that distribution. Similarly, if we assume a Gaussian process, then we can sample functions which satisfy the mean and covariance constraints.

# %% [markdown]

# ## Example:
# > ## Assume a Gaussian Process over the interval $[0,1]$, $f(x) \sim \mathcal{GP}(0, k(x, x'))$ where $k(x,x')$ is the squared exponential (RBF) kernel described previously. We now look at how sample functions can be created:
# >> ## Assume a fixed no.of samples $N$ (say $N=1000$) which is equally distributed over the interval $[0,1]$. Also fix a value for $l$. 
# >> ## Then generate a sample from the multivariate normal distribution for ${\bf x} = [x_1, x_2, \dots, x_N]$ given by $f_{\bf x} \sim \mathcal{N}({\bf 0}, K({\bf x}, {\bf x}))$, where $K({\bf x}, {\bf x})$ is given by
# >> ## $K({\bf x}, {\bf x}) = \left[ \begin{array}{cccc} k(x_1, x_1) & k(x_1, x_2) & \dots & k(x_1, x_N) \\ k(x_2, x_1) & k(x_2, x_2) & \dots & k(x_2, x_N) \\ \vdots & \vdots & \dots & \vdots \\ k(x_N, x_1) & k(x_N, x_2) & \dots & k(x_N, x_N) \\ \end{array} \right]$.

# %%

N = 1000  # no.of sample points
l = 1  # std devation of covariance kernel

# get the points in the interval (0,1)
x = np.linspace(0,1,N)

# get the covariance matrix
K = np.zeros([N,N])
for i in range(N):
     for j in range(N):
          K[i,j] = np.exp(-(x[i]-x[j])*(x[i]-x[j])/(l*l))

# generate samples from the multivariate normal distribution
mean = np.zeros(N)
fx1 = rand.multivariate_normal(mean, K)
fx2 = rand.multivariate_normal(mean, K)
fx3 = rand.multivariate_normal(mean, K)

# plot the sample functions
t1 = go.Scatter(x=x, y=fx1, name='fx1',mode='markers')
t2 = go.Scatter(x=x, y=fx2, name='fx2',mode='markers')
t3 = go.Scatter(x=x, y=fx3, name='fx3',mode='markers')
fig3 = go.Figure(data=[t1, t2, t3])
fig3.update_layout(
     title = 'Sample functions generated for l=1',
     width=800, 
     height=400,
     xaxis_title='x',
     yaxis_title='f(x)',
)

# %%

N = 1000  # no.of sample points
l = 0.1  # std devation of covariance kernel

# get the points in the interval (0,1)
x = np.linspace(0,1,N)

# get the covariance matrix
K = np.zeros([N,N])
for i in range(N):
     for j in range(N):
          K[i,j] = np.exp(-(x[i]-x[j])*(x[i]-x[j])/(l*l))

# generate samples from the multivariate normal distribution
mean = np.zeros(N)
fx1 = rand.multivariate_normal(mean, K)
fx2 = rand.multivariate_normal(mean, K)
fx3 = rand.multivariate_normal(mean, K)

# plot the sample functions
t1 = go.Scatter(x=x, y=fx1, name='fx1',mode='markers')
t2 = go.Scatter(x=x, y=fx2, name='fx2',mode='markers')
t3 = go.Scatter(x=x, y=fx3, name='fx3',mode='markers')
fig3 = go.Figure(data=[t1, t2, t3])
fig3.update_layout(
     title = 'Sample functions generated for l=0.1',
     width=800, 
     height=400,
     xaxis_title='x',
     yaxis_title='f(x)',
)

# %% [markdown]

# # Gaussian Process Regression (GPR)
# ## The idea of GPR is: *Given a prior Gaussian process and a training dataset $(X,{\bf y})$, arrive at a posterior Gaussian process which fits the data well.*
# ## Let $(X,{\bf y})$ denote the training dataset with $X = [{\bf x}_1, {\bf x}_2, \dots, {\bf x}_N]$ and ${\bf y} = [y_1, y_2, \dots, y_N]$ denoting the function values. We want to be able to predict the function values ${\bf y}_*$ at some other points $X_*$ different from the training points.
# ## Prior will be assumed to be $f({\bf x}) \sim \mathcal{GP}({\bf 0}, k({\bf x}, {\bf x'}))$.
# ## The Posterior in terms of values at test points is now given by 
# ---
# > ## ${\bf y}_* \mid {\bf y},X,X_* \sim \mathcal{N}({\bf m}_*, \text{cov}({\bf y}_*))$,
# ---
# > ## ${\bf m}_* = K(X_*,X) K(X,X)^{-1} {\bf y}$,
# > ## $\text{cov}({\bf y}_*) = K(X_*,X_*)-K(X_*,X)K(X,X)^{-1}K(X,X_*)$.

# %% [markdown]

# # GPR with noisy observations
# ## In case of noisy observations
# > ## $y = f({\bf x}) + \epsilon$, $\qquad$ where $\epsilon \sim \mathcal{N}(0,\sigma_n^2)$,
# ## the posterior is given by
# ---
# > ## ${\bf y}_* \mid {\bf y},X,X_* \sim \mathcal{N}({\bf m}_*, \text{cov}({\bf y}_*))$,
# ---
# > ## ${\bf m}_* = K(X_*,X) \left[ K(X,X) + \sigma_n^2 I \right]^{-1} {\bf y}$,
# > ## $\text{cov}({\bf y}_*) = K(X_*,X_*)-K(X_*,X) \left[ K(X,X) + \sigma_n^2 I \right]^{-1} K(X,X_*)$.

# %% [markdown]

# # A practical algorithm

# %%
from IPython.display import Image
Image(filename='algo.jpg',width=1000, height=600)

# %% [markdown]

# # Example

# %%

# loading the dataset
filename = 'Datasets/Dataset 1'
input_df = pd.read_csv(r'{}.csv'.format(filename), low_memory=False)

input_df = input_df.drop(columns=['Unnamed: 0'])
input_df

# %%

X = np.vstack(input_df['x'])
y = np.vstack(input_df['y'])

# %%

#plt.scatter(x=X, y=y)
plt.plot(y, marker='o', linestyle='')
plt.title('Raw data', fontsize=24)

# %% [markdown]

# # Noise-free GPR

# %%

# length of data
N = X.shape[0]

# parameter values to be used for l of the squared exponential kernel
l_list = np.array([1, 0.5])

for i in range(len(l_list)):

     # parameter l of the squared exponential kernel
     l = l_list[i]
     #l=1
     # training K
     K = np.zeros([N,N])
     for i in range(N):
          for j in range(N):
               K[i,j] = np.exp(-(X[i]-X[j])*(X[i]-X[j])/(l*l))
     Kinv = np.linalg.inv(K)

     # test points
     Nt = 2000
     Xt = np.linspace(0,100,Nt)  # 2000 pts evenly distributed b/w 0 and 100

     Ktt = np.zeros([Nt,Nt])
     for i in range(Nt):
          for j in range(Nt):
               Ktt[i,j] = np.exp(-(Xt[i]-Xt[j])*(Xt[i]-Xt[j])/(l*l))

     Kt = np.zeros([Nt,N])
     for i in range(Nt):
          for j in range(N):
               Kt[i,j] = np.exp(-(Xt[i]-X[j])*(Xt[i]-X[j])/(l*l))

     # mean of test set
     m = np.dot(Kt, np.dot(Kinv,y))
     # covariance of test set
     cov = Ktt - np.dot(Kt, np.dot(Kinv, Kt.transpose()))
     std = np.diag(np.sqrt(cov))
     conf1 = m.flatten() - std
     conf2 = m.flatten() + std

     # plot the sample functions
     t = []
     t.append(go.Scatter(x=Xt, y=conf1, name='lower conf. bound', mode='lines', line_color='grey'))
     t.append(go.Scatter(x=Xt, y=conf2, name='upper conf. bound', mode='lines', fill='tonexty', line_color='grey'))
     t.append(go.Scatter(x=Xt, y=m.flatten(), name='mean', mode='markers', marker_size=3))
     t.append(go.Scatter(x=X.flatten(), y=y.flatten(), name='raw data points', mode='markers', marker_size=3, marker_color='red'))

     fig4 = go.Figure(data=t)
     fig4.update_layout(
          title = 'Predictions for l='+str(l),
          width=800, 
          height=400,
          xaxis_title='x',
          yaxis_title='f(x)',
     )
     fig4.show()

# %% [markdown]

# # GPR with noise

# %%

# length of data
N = X.shape[0]

# parameter values to be used for l of the squared exponential kernel - (l,sigma_n)
l_list = np.array([(2, 0.7), (1, 0.7), (0.5, 0.7), (2, 0.3), (1, 0.3), (0.5, 0.3)])

for i in range(len(l_list)):

     # parameter l of the squared exponential kernel
     l = l_list[i][0]
     s = l_list[i][1]  # noise variance
     #l=1
     # training K
     K = np.zeros([N,N])
     for i in range(N):
          for j in range(N):
               K[i,j] = np.exp(-(X[i]-X[j])*(X[i]-X[j])/(l*l))
     K_ = K + s*s*np.eye(N)
     Kinv = np.linalg.inv(K_)

     # test points
     Nt = 2000
     Xt = np.linspace(0,100,Nt)  # 2000 pts evenly distributed b/w 0 and 100

     Ktt = np.zeros([Nt,Nt])
     for i in range(Nt):
          for j in range(Nt):
               Ktt[i,j] = np.exp(-(Xt[i]-Xt[j])*(Xt[i]-Xt[j])/(l*l))

     Kt = np.zeros([Nt,N])
     for i in range(Nt):
          for j in range(N):
               Kt[i,j] = np.exp(-(Xt[i]-X[j])*(Xt[i]-X[j])/(l*l))

     # mean of test set
     m = np.dot(Kt, np.dot(Kinv,y))
     # covariance of test set
     cov = Ktt - np.dot(Kt, np.dot(Kinv, Kt.transpose()))
     std = np.diag(np.sqrt(cov))
     conf1 = m.flatten() - std
     conf2 = m.flatten() + std

     # plot the sample functions
     t = []
     t.append(go.Scatter(x=Xt, y=conf1, name='lower conf. bound', mode='lines', line_color='grey'))
     t.append(go.Scatter(x=Xt, y=conf2, name='upper conf. bound', mode='lines', fill='tonexty', line_color='grey'))
     t.append(go.Scatter(x=Xt, y=m.flatten(), name='mean', mode='markers', marker_size=3))
     t.append(go.Scatter(x=X.flatten(), y=y.flatten(), name='raw data points', mode='markers', marker_size=3, marker_color='red'))

     fig5 = go.Figure(data=t)
     fig5.update_layout(
          title = 'Predictions for s='+str(s)+', l='+str(l),
          width=800, 
          height=400,
          xaxis_title='x',
          yaxis_title='f(x)',
     )
     fig5.show()
