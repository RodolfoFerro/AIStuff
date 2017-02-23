from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
try:
    import seaborn as sns
except:
    print("I wasn't able to import seaborn, but it's okay.")

# Set figure and axis:
fig = plt.figure()
ax  = fig.gca(projection='3d')

# Create data:
# f:
x = np.linspace(0, 5, 1e3)
y = np.linspace(0, 5, 1e3)
x, y = np.meshgrid(x, y)
z = x**2 - 4*x + y**2 - y - x*y
# g:
# x = np.linspace(-1, 1, 1e3)
# y = np.linspace(-1, 1, 1e3)
# x, y = np.meshgrid(x, y)
# z = -np.exp(1. - x**2 - y**2)

# Plot the surface:
surf = ax.plot_surface(x, y, z, cmap=cm.viridis, linewidth=0, antialiased=False)

# Add a color bar which maps values to colors:
fig.colorbar(surf, shrink=0.5, aspect=5)
fig.subplots_adjust(hspace=0.0, wspace=-0.2, top=0.8)

plt.show()
