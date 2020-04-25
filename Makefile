
lint:
	@echo "    Linting irksome codebase"
	@python -m flake8 irksome
	@echo "    Linting irksome test suite"
	@python -m flake8 tests

THREADS=1
ifeq ($(THREADS), 1)
	PYTEST_ARGS=
else
	PYTEST_ARGS=-n $(THREADS)
endif

test: lint
	@echo "    Running all regression tests"
	@python -m pytest tests $(PYTEST_ARGS)
