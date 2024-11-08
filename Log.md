# Log

## Fri 18 Oct

### [Editing Python in Visual Studio Code](https://code.visualstudio.com/docs/python/editing)

- [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort),
  [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
- `numpy` as `np`, `tensorflow` as `tf`, `pandas` as `pd`, `matplotlib.pyplot`
  as `plt`, `matplotlib` as `mpl`, `math` as `m`, `scipi.io` as `spio`, and
  `scipy` as `sp`, `panel` as `pn`, `holoviews` as `hv`

- `> Organize Imports`
- Keys
  - <kbd>F12</kbd> — Go to Definition
  - <kbd>⌥F12</kbd> — Peek definition

- [PyLance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
  — language server

- [Pyright](https://github.com/microsoft/pyright) — static type checker

  - [Pyright playground](https://pyright-play.net/?code=MQAgKgFglgziMEMC2AHANgUxAEw0g9gHYwAuATgiRnBPgO4gDG%2BSBhIGZZ%2BZcjC7AEZZcVRlWzwSlKPzRoAniEFKUCslADmEEgDoAUPtwAzEAmzYAFAA8AXCGNp8lADQgF9x85IBKW-pBAkDIMEgBXMnZrEABqd0NQAAUEGBgoQk0zKTIQdNIBRiwUkBIILBgMZkJJBDJNMKQMQhJg6jC0Ejh0rLIw5qhGjmtClBIoIgNzKwBGNwAiOZ99IA)
  - [Stub Files](https://typing.readthedocs.io/en/latest/spec/distributing.html#stub-files)
    (?) — .pyi
  - [Writing and Maintaining Stub Files](https://typing.readthedocs.io/en/latest/guides/writing_stubs.html#writing-stubs)

- [VS Code's Python Extension](https://code.visualstudio.com/docs/python/editing)
  - [Code analysis settings](https://code.visualstudio.com/docs/python/settings-reference#_code-analysis-settings)
  - [Autocomplete settings](https://code.visualstudio.com/docs/python/settings-reference#_autocomplete-settings)
  - Provides autocomplete and IntelliSense for all files within the current
    working folder
  - It counts how often you use certain symbols
  - Enables a minimum set of features
  - Customizable analysis engine
  - Auto imports: `python.analysis.autoImportCompletions: true` (disabled by
    default)
  - Custom package locations:

```python
"python.analysis.extraPaths": [
    "~/.local/lib/Google/google_appengine",
    "~/.local/lib/Google/google_appengine/lib/flask-0.12"
]
```

- Copilot ad

### [importResolveFailure](https://code.visualstudio.com/docs/python/editing#_importresolvefailure)

This error happens when Pylance is unable to find the package or module you're
importing, nor its type stubs.

**How to fix it**

- If you are importing a module, make sure it exists in your workspace or in a
  location that is included in the `python.autoComplete.extraPaths` setting.
- If you are importing a package that is not installed, you can install it by
  running the following command in an activated terminal:
  `python -m pip install {package_name}`.
- If you are importing a package that is already installed in a different
  interpreter or kernel,
  [select the correct interpreter](https://code.visualstudio.com/docs/python/environments#_select-and-activate-an-environment).
- If you are working with an editable install and it is currently set up to use
  import hooks, consider switching to using `.pth` files that only contain file
  paths instead, to enhance compatibility and ensure smoother import behavior.
  [Learn more in the Pyright documentation](https://microsoft.github.io/pyright/#/import-resolution?id=editable-installs).

> Мне кажется, установка Pylance решила все мои проблемы.

## Sun 20 Oct

### [Значение расширения .pyi в Python и его содержимого](https://ru.stackoverflow.com/questions/1012021/%D0%97%D0%BD%D0%B0%D1%87%D0%B5%D0%BD%D0%B8%D0%B5-%D1%80%D0%B0%D1%81%D1%88%D0%B8%D1%80%D0%B5%D0%BD%D0%B8%D1%8F-pyi-%D0%B2-python-%D0%B8-%D0%B5%D0%B3%D0%BE-%D1%81%D0%BE%D0%B4%D0%B5%D1%80%D0%B6%D0%B8%D0%BC%D0%BE%D0%B3%D0%BE)

`.pyi` файлы - это стабы (stubs), их назначение и формат описаны в PEP 484. Эти
файлы вообще никак не используются интерпретатором, их назначение -
предоставлять информацию о типизации кода. К примеру, у тебя есть сторонний
модуль без типизации, который не твой, редактировать его ты не можешь, а
прописать типы хотелось бы:

```python
# fizz.py
def greet(who):
    return f'Hello {who}'
```

Выход находится с помощью стаба: создаешь файл `fizz.pyi`, который содержит
типизированную сигнатуру `greet` без имплементации:

```python
# fizz.pyi
def greet(who: str) -> str:
    ...
```

Теперь утилиты статической типизации типа `mypy` или автодополнение кода в
Pycharm знают, где найти типизацию для функции `greet`.

### Что-то про Torch

```python
>>> torch.ones(8)
tensor([1., 1., 1., 1., 1., 1., 1., 1.])
```

```python
>>> torch.bernoulli(torch.ones(8) * 0.5)
tensor([1., 0., 1., 1., 1., 0., 1., 1.])
```

```python
>>> torch.bernoulli(torch.ones(8) * 0.5)
tensor([1., 0., 1., 1., 0., 1., 1., 0.])
```

```python
>>> torch.bernoulli(torch.ones(8) * 0.1)
tensor([0., 0., 0., 0., 0., 0., 0., 0.])
```

```python
>>> torch.bernoulli(torch.ones(8) * 0.1)
tensor([0., 1., 0., 1., 0., 0., 0., 0.])
```

### [Tensor.expand](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html)

```python
Tensor.expand(*sizes) → Tensor
```

Returns a new view of the `self` tensor with singleton dimensions expanded to a
larger size.

```python
>>> x = torch.tensor([[1], [2], [3]])
>>> x.size()
torch.Size([3, 1])
>>> x.expand(3, 4)
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
>>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
```

### [torch.unsqueeze](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html)

```python
torch.unsqueeze(input, dim) → Tensor
```

Returns a new tensor with a dimension of size one inserted at the specified
position.

```python
>>> x = torch.tensor([1, 2, 3, 4])
>>> torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
>>> torch.unsqueeze(x, 1)
tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])
```

### [torch.squeeze](https://pytorch.org/docs/stable/generated/torch.squeeze.html#torch.squeeze)

```python
torch.squeeze(input, dim: int | tuple[int, int, ...int] = None) → Tensor
```

Returns a tensor with all specified dimensions of input of size 1 removed.

For example, if _input_ be of shape
$(A \times 1 \times B \times C \times 1 \times D)$, then output be of shape
$(A \times B \times C \times D)$.

**BUT** if `dim` is given, then a squeeze operation is done only in the given
dimension(s).

Example:

```python
>>> x = torch.zeros(2, 1, 2, 1, 2)
>>> x.size()
torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x)
>>> y.size()
torch.Size([2, 2, 2])
>>> y = torch.squeeze(x, 0)
>>> y.size()
torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x, 1)
>>> y.size()
torch.Size([2, 2, 1, 2])
>>> y = torch.squeeze(x, (1, 2, 3))
torch.Size([2, 2, 2])
```

### `*.__class__.__name__.` в Python

Эта красивая штука работает вообще для всего:

```python
>>> (1).__class__.__name__
'int'
>>> (None).__class__.__name__
'NoneType'
>>> (str).__class__.__name__
'type'
>>> (print).__class__.__name__
'builtin_function_or_method'
```

Текст можно писать прямо в превью!

- [PrinceXML](https://www.princexml.com/)
- [Bibliographies and citations](https://shd101wyy.github.io/markdown-preview-enhanced/#/pandoc-bibliographies-and-citations)

```yaml
---
title: "Important document"
output: pdf_document
bibliography: bibliography.bib
# Или даже:
bibliography: [bibliography1.bib, bibliography2.bib]
---
```

Поддерживается:

| Format      | File extension |
| ----------- | -------------- |
| BibLaTeX    | .bib           |
| BibTeX      | .bibtex        |
| Copac       | .copac         |
| CSL JSON    | .json          |
| CSL YAML    | .yaml          |
| EndNote     | .enl           |
| EndNote XML | .xml           |
| ISI         | .wos           |
| MEDLINE     | .medline       |
| MODS        | .mods          |
| RIS         | .ris           |

### Converting back and forth between .py and .ipynb

```bash
(.venv) 127     JupyterNotebooksλ ~/.venv/bin/jupytext --to notebook hypertrain5.py
[jupytext] Reading hypertrain5.py in format py
[jupytext] Writing hypertrain5.ipynb (destination file replaced [use --update to preserve cell outputs and ids])

$ jupyter nbconvert --to script hypertrain5.ipynb
```

### Расширяем diagonal_mask

```python
diagonal_mask = diagonal_mask.expand(grad_output.size(1), -1).t()
```

```
Exception has occurred: RuntimeError
The expanded size of the tensor (1024) must match the existing size (12) at non-singleton dimension 0.  Target sizes: [1024, -1].  Tensor sizes: [12, 12]
  File "/Users/scales/JupyterNotebooks/hypertrain5.py", line 370, in backward
    diagonal_mask = diagonal_mask.expand(grad_output.size(1), -1).t()
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/scales/JupyterNotebooks/hypertrain5.py", line 520, in train
    loss.backward()
  File "/Users/scales/JupyterNotebooks/hypertrain5.py", line 608, in <module>
    train(
RuntimeError: The expanded size of the tensor (1024) must match the existing size (12) at non-singleton dimension 0.  Target sizes: [1024, -1].  Tensor sizes: [12, 12]
```

:c

Пробуем что-то другое

```
diagonal_mask = diagonal_mask.unsqueeze(1).expand(-1, grad_output.size(1), -1)
diagonal_mask = diagonal_mask.permute(0, 2, 1)

grad_output = grad_output.unsqueeze(1) * diagonal_mask
grad_output = grad_output.sum(dim=1)
```

Запускаем

```
grad_output shape: torch.Size([12, 1024])
diagonal_mask shape: torch.Size([12, 12, 1024])

Exception has occurred: RuntimeError
mat2 must be a matrix
  File "/Users/scales/JupyterNotebooks/hypertrain5.py", line 380, in backward
    grad_output = grad_output.mm(diagonal_mask)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/scales/JupyterNotebooks/hypertrain5.py", line 525, in train
    loss.backward()
  File "/Users/scales/JupyterNotebooks/hypertrain5.py", line 613, in <module>
    train(
RuntimeError: mat2 must be a matrix
```

Одно измерение лишнее…

Там была одна лишняя строчка, попробую ещё раз

![Средние значения градиента по итерациям](img/AbsMeanGradient.png)
![Кривая потерь](img/Loss.png)

![Среднеквадратичное отклонение](img/MetricsVsNoise.png)
![R^2](img/MetricsVsNoise2.png)

По сравнению с предыдущей итерацией:

- MSE падает, а не растёт
- $R^2$ растёт, а не падает

По определению
([Коэффициент детерминации](https://ru.wikipedia.org/wiki/%D0%9A%D0%BE%D1%8D%D1%84%D1%84%D0%B8%D1%86%D0%B8%D0%B5%D0%BD%D1%82_%D0%B4%D0%B5%D1%82%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D0%B0%D1%86%D0%B8%D0%B8)):

$$
R^2 = 1 - \frac{D[y|x]}{D[y]}
$$

- $D[y] = \sigma_y^2$ — дисперсия случайной величины (разницы между фактическими
  и предсказанными значениями)
- $D[y|x] = \sigma^2$ — дисперсия ошибки модели

План дальше:

- [x] Поменять гиперпараметры??
- [ ] Проверить на новых данных (_??_)
- [ ] Анализ графика регрессионных остатков (_??_)

## Fri 8 Nov

> 'Axes' object has no attribute 'xlabel'
> 'Axes' object has no attribute 'xscale'

### [matplotlib.axes.Axes](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes)

```python
def matplotlib.axes.Axes(
  fig,
  *args,
  facecolor=None,
  frameon=True,
  sharex=None,
  sharey=None,
  label='',
  xscale=None,
  yscale=None,
  box_aspect=None,
  forward_navigation_events='auto',
  **kwargs
)
```

> An Axes object encapsulates all the elements of an individual (sub-)plot in a
> figure.
>
> It contains most of the (sub-)plot elements: Axis, Tick, Line2D, Text,
> Polygon, etc., and sets the coordinate system.
>
> Like all visible elements in a figure, Axes is an Artist subclass.

То есть _Axes_ это такой кусок графика… Поэтому у него нет `.set_xscale()`.

Потому что скорее всего `.set_xscale()` мы будем вызывать на `fig`?

### [matplotlib.pyplot.subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)

```python
matplotlib.pyplot.subplots(
  nrows=1, 
  ncols=1, 
  *, 
  sharex=False, sharey=False, squeeze=True, width_ratios=None, 
  height_ratios=None, subplot_kw=None, gridspec_kw=None, **fig_kw
) -> (Figure, Axes)
```

### [matplotlib.figure.Figure](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure)

```python
class matplotlib.figure.Figure(
  figsize=None, dpi=None, *,
  facecolor=None, edgecolor=None, linewidth=0.0, frameon=None, 
  subplotpars=None, tight_layout=None, constrained_layout=None, 
  layout=None, **kwargs
)
```

> The top level container for all the plot elements.
>
> See [matplotlib.figure](https://matplotlib.org/stable/api/figure_api.html#module-matplotlib.figure) for an index of class methods.

### [matplotlib.figure](https://matplotlib.org/stable/api/figure_api.html#module-matplotlib.figure)

> 'Axes' object has no attribute 'xlabel'
> 'Axes' object has no attribute 'xscale'

- Мне нужен Figure.savefig?

- Мне точно нужны [matplotlib.axes.Axes.set_xlabel](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html#matplotlib.axes.Axes.set_xlabel)
  и [matplotlib.axes.Axes.set_xscale](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html#matplotlib.axes.Axes.set_xscale)

### [matplotlib.artist](https://matplotlib.org/stable/api/artist_api.html#matplotlib.artist.Artist)

```
Artist.Tick.XTick.ThetaTick
      :    .YTick.RadialTick
      .Patch.FancyArrowPatch.ConnectionPatch
      :     .FancyBboxPatch
      .Collection
```