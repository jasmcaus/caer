## Type Checking
#### Static type checking and real-time linting for Vision code.

By this point, you've probably seen that ``Caer`` uses the new **Python 3.6+** syntax for [type hints](https://docs.python.org/3/library/typing.html) (also known as *type annotations*). The entire code base is type-annotated and it is recommended that you add *at least some types* to your own code, too! 

Type annotations can make your numeric code much more explicit, making it (much) easier to debug. They also allow your editor (and other tools) to perform type checks before executing your code. For example, if you try to add a ``str`` and an ``int``, your editor will probably warn you that it is an invalid operation without having to wait until you run the invalid code. It may also tell you that a function expects a ``float``, so you don’t pass it an invalid type. If your layer is typed as ``caer.Tensor``, Caer can warn you if its inputs and outputs are incompatible with the rest of your Vision model.


Our’s type-system won’t catch every error: it has no representation for the sizes of your ``Tensor`` dimensions, so a lot of invalid operations can’t be detected until runtime. Sometimes the syntax gets quite ugly, and the error messages are often frustratingly opaque. Nevertheless, we do recommend you try it out, especially for your model definitions and the functions you end up using.


### Installation & Setup

``mypy`` is the *standard* type-checker for Python. You can install it via ``pip`` or ``conda``. If you use a virtual environment for your project, ensure that you install it in the same environment.

```shell
$ pip install mypy
```

or 

```shell
$ conda install -c conda-forge mypy
```

We are working on a ``mypy`` plugin that will extend the normal functionality of ``mypy`` to perform additional type checks in code using Caer. If you installed Caer, you already have the plugin. To enable the Caer plugin for ``mypy``, you just have to create a file ``mypy.ini`` at the *root* of your project folder. This will tell ``mypy`` to use the plugin in the module ``caer.mypy``. If you use ``pydantic`` for advanced configuration, you can also enable ``pydantic``’s plugin. If you’re using Caer as part of your Python package, you can also add the ``[mypy]`` section to your package’s ``setup.cfg``.

```python
[mypy]
plugins = caer.mypy
```

To type check a file or directory, you can now use the ``mypy`` command:

```shell
$ mypy file.py
```

### Setting up linting in your editor

Real-time linting is especially powerful, as it lets you type-check your code as it leaves your fingers. This often lets you catch errors in their original context, when they’re least confusing. It can also save you trips to the documentation.


If you use [Visual Studio Code](https://code.visualstudio.com/), make sure you install the [Python extension](https://code.visualstudio.com/docs/python/python-tutorial#_install-visual-studio-code-and-the-python-extension). Then select the appropriate [environment](https://code.visualstudio.com/docs/python/environments) in your editor. If you installed mypy in the same environment and select it in your editor, after adding the mypy.ini file (as described above) everything should work.


For [PyCharm](https://www.jetbrains.com/pycharm/) users, make sure you [configure the Python Interpreter](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html) for your project. Then install the “Mypy” plugin. You may also want to install the [“Mypy ​(Official)” plugin](https://plugins.jetbrains.com/plugin/11086-mypy). If you installed mypy in the same environment/interpreter, after adding the mypy.ini file (as described above) and installing the plugin, everything should work.


For other editors, the [``mypy`` docs](https://github.com/python/mypy#ide-linter-integrations-and-pre-commit) has instructions for editors like Vim, Emacs, Sublime Text and Atom.


### Static Type Checking

*Static type checking* means that your editor (or other tools) will check the code using the declared types before running it. Because it is done before running the code, it’s called “static”. The contrary would be *dynamic* type checking, where checks are performed at runtime, while the program is running and the code is being executed. (Once complete, Caer will also do runtime validation!) As editors and similar tools can’t just randomly run your code to verify that it’s correct, we have these type annotations to help editors check the code and provide autocompletion.


Even if you never run a type-checker, adding type-annotations to your code can greatly improve its readability. Multi-dimensional Tensor libraries like ``coreten`` and ``numpy`` make it easy to write terse, fairly general code; but when you revisit the code later, it’s often very hard to figure out what’s happening without executing the code and debugging.

Consider this function (from the ``caer.transforms`` module): 

```python
def darken(img: Tensor, coeff: float) -> Tensor:
    ... 
    return darkened
```

over 

```python
def darken(img, coeff):
    ... 
    return darkened
```

Type annotations provide a relatively concise way to document some of the most important information about your code. The same information can be provided in comments, but unless you use consistent syntax, your type comments will probably be much longer and more distracting than the equivalent annotations.


Another advantage of type annotations as documentation is that they can be queried for more detail, while with comments, you have to choose the level of detail to provide up-front. Thinc’s type annotations take into account numpy’s tricky indexing system, and also the semantics of the different reduction operations as different arguments are passed in. This makes it much easier to follow along with steps that might have felt obvious to the author of the code.

