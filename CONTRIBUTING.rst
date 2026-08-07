============
Contributing
============

Please open an issue before substantial changes. Keep pull requests focused,
add tests for changed behavior, and ensure these commands pass::

    poetry install
    poetry run coverage run -m pytest
    poetry run coverage report --fail-under=80
    poetry run ruff check imcluster tests
    poetry run ruff format --check imcluster tests
    poetry run mypy imcluster
    poetry run sphinx-build -W -b html docs docs/_build/html
    poetry build

Do not commit model weights, generated reports, caches, or private image data.
Contributions are accepted under the Apache License 2.0.
