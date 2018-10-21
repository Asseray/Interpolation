# Newton Interpolation Polynomial

import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as ax_ar
import numpy as np


def diff_quo(y1, y2, x1, x2):
    # diff_quo: difference quotient
    return (y1-y2)/(x1-x2)


def renew_line(last_line, x_arr_ahead, y):
    # last_line: {x_n, f(x_n), f[x_(n-1), x_n], ...},
    # x_arr_ahead: {x_0, x_1, ..., x_n+1}, y: f(x_n+1)
    # return new_line
    new_line = np.array([x_arr_ahead[-1], y])
    for n in range(1, last_line.size):
        new_diff_quo = diff_quo(new_line[n], last_line[n], new_line[0], x_arr_ahead[-n-1])
        new_line = np.append(new_line, new_diff_quo)
    return new_line


def dq_producer(x_arr, y_arr):
    # 'dq' denotes "difference quotient".
    # x_arr: {x_0, x_1, ...},
    # y_arr: {f(x_0), f(x_1), ...}
    # return dq_arr: {f[x_0], f[x_0, x_1], f[x_0, x_1, x_2], ...}
    if x_arr.size != y_arr.size:
        raise Exception('x_arr.size != y_arr.size')
    dq_arr = np.array([y_arr[0]])
    last_line = np.array([x_arr[0], y_arr[0]])
    for n in range(1, x_arr.size):
        last_line = renew_line(last_line, x_arr[:n+1], y_arr[n])
        dq_arr = np.append(dq_arr, last_line[-1])
    return dq_arr


def pol_value(dq_arr, x_arr, x):
    # dq_arr: {f[x_0], f[x_0, x_1], f[x_0, x_1, x_2], ...}
    # x: np.array(list of values of variable)
    # return y, (y: list of values of interpolation function at corresponding nodes)
    m = x.size
    y = np.array([dq_arr[0]]*m, dtype='float64')
    x_product = np.array([1]*m, dtype='float64')
    for n in range(1, x_arr.size):
        x_product *= x - x_arr[n-1]
        y += dq_arr[n]*x_product
    return y


# ++++++++++++++++++++++++ config area +++++++++++++++++++++++++++
def f(x):
    # the initial function
    return np.sin(x)


f_text = 'f(x) = sin(x)'  # used in legend
num_pl_node = 5  # number of nodes of interpolation
pol_interval = np.array([0, 2*np.pi])  # interpolation interval
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# calculate the values of interpolation function
x_node = np.linspace(pol_interval[0], pol_interval[1], num_pl_node)  # nodes of interpolation
y_value = f(x_node)  # {f(x_i)}
x_plot = np.linspace(x_node.min(), x_node.max(), num=100)  # points for plot on the x-axis
dq = dq_producer(x_node, y_value)  # an array of difference quotients
pv_plot = pol_value(dq, x_node, x_plot)  # values of interpolation function at points for plot


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
plt.plot(x_plot, pv_plot,
         color='red',
         linestyle='--',
         label='P(x)')
plt.scatter(x_node, y_value)
plt.legend([f_text, 'P(x)', 'nodes of interpolation'])

plt.show()
