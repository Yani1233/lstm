# Default target executed when no arguments are given to make.
all: help

######################
# TESTING AND COVERAGE
######################

# Define a variable for the test file path.
TEST_FILES ?= tests/
coverage test tests: TEST_FILES=tests/
coverage_diff test_diff tests_diff: TEST_FILES=$(shell git diff --name-only --diff-filter=d develop | grep -E 'tests/.*\.py$$')

# Run unit tests and generate a coverage report.
coverage:
	pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered \
		$(TEST_FILES)

coverage_diff:
	[ "$(TEST_FILES)" = "" ] || pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered \
		$(TEST_FILES)


test tests:
	pytest --disable-socket --allow-unix-socket $(TEST_FILES)

test_diff tests_diff:
	[ "$(TEST_FILES)" = "" ] || pytest --disable-socket --allow-unix-socket $(TEST_FILES)

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
lint format: PYTHON_FILES=.
lint_diff format_diff spell_check_diff spell_fix_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d develop | grep -E '\.py$$|\.ipynb$$')

lint lint_diff:
	[ "$(PYTHON_FILES)" = "" ] || flake8 $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mypy --ignore-missing-imports --disallow-untyped-defs $(PYTHON_FILES)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || black --config=./.black $(PYTHON_FILES)

spell_check:
	codespell

spell_fix:
	codespell -w

spell_check_diff:
	[ "$(PYTHON_FILES)" = "" ] || codespell $(PYTHON_FILES)

spell_fix_diff:
	[ "$(PYTHON_FILES)" = "" ] || codespell -w $(PYTHON_FILES)

######################
# HELP
######################

help:
	@echo '===================='
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'spell_check               	- run codespell on the project'
	@echo 'spell_fix               		- run codespell on the project and fix the errors'
	@echo '-- TESTS --'
	@echo 'coverage                     - run unit tests and generate coverage report'
	@echo 'tests                        - run unit tests (alias for "make test")'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo '-- DOCUMENTATION tasks are from the top-level Makefile --'
