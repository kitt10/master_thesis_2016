import matplotlib.pyplot as plt
import numpy as np
from kitt_nn.nn_tool.nn_function import sigmoid, tanh

x = np.arange(-100, 100, 0.01)
plt.plot(x, sigmoid(x), label='f(z): sigmoid')
plt.plot(x, tanh(x), label='tanh(z)')
plt.xlabel('z')
plt.xlim([-15, 15])
plt.ylim([-1.5, 1.5])
plt.grid()
plt.legend(loc='best')
plt.savefig('../../thesis/img/transfer_functions.eps', bbox_inches = 'tight', pad_inches = 0.1)
plt.show()