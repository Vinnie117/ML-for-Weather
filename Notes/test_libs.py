######## Python libs in Visual Studio COde
#
#### In Powershell Terminal
# create venv
# py -3 -m venv .venv

# temporarily change the PowerShell execution policy to allow scripts to run
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# activate venv
# .venv\scripts\activate

# Install Libs
# python -m pip install matplotlib

# List of installed Libs
# pip list
####

import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()   