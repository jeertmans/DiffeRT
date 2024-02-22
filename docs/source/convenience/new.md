(convenience:new)=

# Making a new convenience package

There is a [cookiecutter](https://github.com/cookiecutter/cookiecutter)
template provided in this repo that can be used to make new packages.

```bash
pip install cookiecutter
cookiecutter contrib/template -o contrib/
```

This should ask you a bunch of questions, and generate a directory
named after your project with a python package. From there, you should:

1. Edit the `__init__.py` file to fill in the command used to start your
   process, any environment variables, and title of the launcher icon.
2. (Optionally) Add a square svg icon for your launcher in the `icons`
   subfolder, with the same name as your project.
