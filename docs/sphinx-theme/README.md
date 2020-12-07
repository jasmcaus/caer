# Caer Sphinx Theme

Sphinx theme for [Caer Docs](https://caer.readthedocs.io/en/latest/) based on the [Read the Docs Sphinx Theme](https://sphinx-rtd-theme.readthedocs.io/en/latest).

Adapted from the [PyTorch Sphinx Theme](https://github.com/pytorch/pytorch_sphinx_theme).

## Local Development

Run python setup:

```bash
python setup.py install
```


### Built-in Stylesheets and Fonts

There are a couple of stylesheets and fonts inside the Docs and Tutorials repos themselves meant to override the existing theme. To ensure the most accurate styles we should comment out those files until the maintainers of those repos remove them:

#### Docs

```bash
# ./docs/source/conf.py

html_context = {
    # 'css_files': [
    #     'https://fonts.googleapis.com/css?family=Lato',
    #     '_static/css/pytorch_theme.css'
    # ],
}
```

#### Tutorials

```bash
# ./conf.py

# app.add_stylesheet('css/pytorch_theme.css')
# app.add_stylesheet('https://fonts.googleapis.com/css?family=Lato')
```


### Top/Mobile Navigation

The top navigation and mobile menu expect an "active" state for one of the menu items. To ensure that either "Docs" or "Tutorials" is marked as active, set the following config value in the respective `conf.py`, where `{project}` is either `"docs"` or `"tutorials"`.

```bash
html_theme_options = {
  ...
  'caer_project': {project}
  ...
}
```

