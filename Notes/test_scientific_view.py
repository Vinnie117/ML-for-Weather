######## Test scientific view in Visual Studio Code

#### Problem: how to get scientific view in VS Code 
# See Github issues 
# https://github.com/microsoft/vscode-python/issues/10559
# https://github.com/microsoft/vscode-jupyter/issues/1286

# Solution
# https://devblogs.microsoft.com/python/python-in-visual-studio-code-january-2021-release/#data-viewer-when-debugging 
# - Set breakpoints
# - Go to debugger view -> right-click dataframe -> View value in Data Viewer

#### Problem: DataViewer in Debugger is buggy with multiple breakpoints
# https://github.com/microsoft/vscode-jupyter/issues/4065
# https://github.com/microsoft/vscode-jupyter/issues/6705

# Fix
# https://github.com/microsoft/vscode-jupyter/issues/5627
# https://github.com/microsoft/vscode-jupyter/pull/5625 
# (just watch the breakpoints)




import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))

print("END")