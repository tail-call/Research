# Log

## Fri 18 Oct

### [Editing Python in Visual Studio Code](https://code.visualstudio.com/docs/python/editing)

- [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort), [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
- `numpy` as `np`, `tensorflow` as `tf`, `pandas` as `pd`,
  `matplotlib.pyplot` as `plt`, `matplotlib` as `mpl`, `math` as `m`,
  `scipi.io` as `spio`, and `scipy` as `sp`, `panel` as `pn`,
  `holoviews` as `hv`

- `> Organize Imports`
- Keys
    - <kbd>F12</kbd> — Go to Definition
    - <kbd>⌥F12</kbd> — Peek definition

- [PyLance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) — language server

- [Pyright](https://github.com/microsoft/pyright) — static type checker

  - [Pyright playground](https://pyright-play.net/?code=MQAgKgFglgziMEMC2AHANgUxAEw0g9gHYwAuATgiRnBPgO4gDG%2BSBhIGZZ%2BZcjC7AEZZcVRlWzwSlKPzRoAniEFKUCslADmEEgDoAUPtwAzEAmzYAFAA8AXCGNp8lADQgF9x85IBKW-pBAkDIMEgBXMnZrEABqd0NQAAUEGBgoQk0zKTIQdNIBRiwUkBIILBgMZkJJBDJNMKQMQhJg6jC0Ejh0rLIw5qhGjmtClBIoIgNzKwBGNwAiOZ99IA)
  - [Stub Files](https://typing.readthedocs.io/en/latest/spec/distributing.html#stub-files) (?) — .pyi
  - [Writing and Maintaining Stub Files](https://typing.readthedocs.io/en/latest/guides/writing_stubs.html#writing-stubs)

- [VS Code's Python Extension](https://code.visualstudio.com/docs/python/editing)
  - [Code analysis settings](https://code.visualstudio.com/docs/python/settings-reference#_code-analysis-settings)
  - [Autocomplete settings](https://code.visualstudio.com/docs/python/settings-reference#_autocomplete-settings)
  - Provides autocomplete and IntelliSense for all files within the
    current working folder
  - It counts how often you use certain symbols
  - Enables a minimum set of features
  - Customizable analysis engine
  - Auto imports: `python.analysis.autoImportCompletions: true` (disabled by default)
  - Custom package locations:

```python
"python.analysis.extraPaths": [
    "~/.local/lib/Google/google_appengine",
    "~/.local/lib/Google/google_appengine/lib/flask-0.12"
]
```


- Copilot ad


### [importResolveFailure](https://code.visualstudio.com/docs/python/editing#_importresolvefailure)

This error happens when Pylance is unable to find the package or module you're importing, nor its type stubs.

**How to fix it**
- If you are importing a module, make sure it exists in your workspace
  or in a location that is included in the `python.autoComplete.extraPaths` setting.
- If you are importing a package that is not installed, you can install it by running the following command in an activated terminal: `python -m pip install {package_name}`.
- If you are importing a package that is already installed in a different interpreter or kernel, [select the correct interpreter](https://code.visualstudio.com/docs/python/environments#_select-and-activate-an-environment).
- If you are working with an editable install and it is currently set up to use import hooks, consider switching to using `.pth` files that only contain file paths instead, to enhance compatibility and ensure smoother import behavior. [Learn more in the Pyright documentation](https://microsoft.github.io/pyright/#/import-resolution?id=editable-installs).
