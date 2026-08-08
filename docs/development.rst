===========
Development
===========

Install and verify the project with:

.. code-block:: bash

   poetry install
   poetry run coverage run -m pytest
   poetry run coverage report --fail-under=80
   poetry run ruff check imcluster tests
   poetry run mypy imcluster
   poetry run sphinx-build -W -b html docs docs/_build/html
   poetry build

See the repository's ``CONTRIBUTING.rst`` and ``SECURITY.md`` files for project
policies. Issues may be reported at
https://github.com/rbturnbull/imcluster/issues.
