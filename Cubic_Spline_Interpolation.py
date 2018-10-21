# Cubic Spline Interpolation

import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as ax_ar
import numpy as np


def delta(x_arr):
    h_arr = x_arr[1:] - x_arr[:-1]
    return h_arr


def diff_quo(y1, y2, x1, x2):
    # diff_quo: difference quotient
    return (y1-y2)/(x1-x2)


def sec_diag(h_arr):
    # return elements on the secondary diagonal of the tridiagonal matrix
    sd_L = h_arr[:-1] / (h_arr[1:] + h_arr[:-1])  # lower secondary diagonal
    sd_u = h_arr[1:]/(h_arr[1:] + h_arr[:-1])  # upper secondary diagonal
    return sd_L, sd_u


def force_term(x_arr, y_arr):
    # Return {F_1, ..., F_(n-1)}. F is a term in AX=F.
    h_arr = delta(x_arr)
    dq2 = delta(y_arr)/h_arr
    dq3 = delta(dq2)/(h_arr[:-1] + h_arr[1:])
    return 6*dq3


def s(m_0, m_1, x_0, x_1, y_0, y_1, x):
    # return sv, (sv: list of values of interpolation function at corresponding nodes)
    f1 = lambda a, b, c, d: a*(b-c)**3/(6*d)
    f2 = lambda a, b, c, d, e: (a-b*c**2/6)*(d-e)/c
    h = x_1-x_0
    sv = f1(m_0, x_1, x, h) + f1(m_1, x, x_0, h)\
        + f2(y_0, m_0, h, x_1, x) + f2(y_1, m_1, h, x, x_0)
    return sv


# ++++++++++++++++++++++++ config area +++++++++++++++++++++++++++
def f(x):
    # the initial function
    return 1/(1+x**2)


f_text = 'f(x) = 1/(1+x²)'  # used in legend
x_node = np.linspace(-5, 5, num=11)  # nodes of interpolation
y_value = f(x_node)  # {f(x_i)}
boundary_cond = np.array([10/(1+25)**2, -10/(1+25)**2])  # the boundary condition, {f'(x_0), f'(x_n)}
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# add the first kind of boundary conditions and generate tridiagonal matrix
delta_x = delta(x_node)  # {x_1-x_0, x_2-x_1, ...}
l_subdiag, u_subdiag = sec_diag(np.concatenate(([0], delta_x, [0]), axis=0))  # secondary diagonal
f_x0x1 = diff_quo(y_value[1], y_value[0], x_node[1], x_node[0])  # f[x_0, x_1]
f_xnr1xn = diff_quo(y_value[-1], y_value[-2], x_node[-1], x_node[-2])  # f[x_n-1, x_n]
f_0 = 6/delta_x[0]*(f_x0x1 - boundary_cond[0])  # F_0
f_n = 6/delta_x[-1]*(boundary_cond[1] - f_xnr1xn)  # F_n
f_term = np.concatenate([[f_0], force_term(x_node, y_value), [f_n]], axis=0)  # F, the term in AX=F.


# solve the tridiagonal matrix by chasing method
beta_new, m_new = u_subdiag[1]/2, f_term[0]/2
beta, m = beta_new, m_new
for n in range(1, x_node.size):
    divisor = 2-l_subdiag[n]*beta_new
    beta_new = u_subdiag[n]/divisor
    m_new = (f_term[n]-l_subdiag[n]*m_new)/divisor
    beta = np.append(beta, beta_new)
    m = np.append(m, m_new)
for n in range(2, x_node.size+1):
    m[-n] = m[-n] - beta[-n]*m[-n+1]


# calculate the values of interpolation function
sv_plot = np.array([])  # values of interpolation function at points for plot
x_plot = np.array([])  # points for plot on the x-axis
for n in range(x_node.size-1):
    x_L, x_r = x_node[n], x_node[n+1]
    x_plot_new = np.linspace(x_L, x_r, num=20)  # num: the number of points for plot in this interval
    sv_plot_new = s(m[n], m[n+1], x_L, x_r, y_value[n], y_value[n+1], x_plot_new)
    x_plot = np.append(x_plot, x_plot_new)
    sv_plot = np.append(sv_plot, sv_plot_new)

# config the figure
fig = plt.figure(num=1)
ax = ax_ar.Subplot(fig, 111)
fig.add_axes(ax)
ax.axis["bottom"].set_axisline_style("->", size=1.5)
ax.axis["left"].set_axisline_style("->", size=1.5)
ax.axis["top"].set_visible(False)
ax.axis["right"].set_visible(False)
plt.xlabel('x')
plt.ylabel('y')

# draw figures
plt.plot(x_plot, f(x_plot),
         label='f(x)')
plt.plot(x_plot, sv_plot,
         color='red',
         linestyle='--',
         label='S(x)')
plt.scatter(x_node, y_value)
plt.legend([f_text, 'S(x)', 'nodes of interpolation'])

plt.show()
