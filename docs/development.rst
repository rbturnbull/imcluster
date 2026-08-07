===========
Development
===========

Install and verify the project with::

   poetry install
   poetry run pytest
   poetry run ruff check imcluster tests
   poetry run mypy imcluster
   poetry run sphinx-build -W -b html docs docs/_build/html
   poetry build

See the repository's ``CONTRIBUTING.rst`` and ``SECURITY.md`` files for project
policies.
