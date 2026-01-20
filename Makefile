# Define the directories and files to remove
DB_DIR = .db
CHROMA_DB_DIR = db
CACHE_DIR = photos/cache
PYTHON = python3
MAIN = main.py
STREAMLIT = streamlit
PYTEST = python -m pytest
TEST_DIR = tests

.PHONY: clean help test test-v test-cov test-file test-case test-match

# The default action if you just type 'make'
help:
	@echo "Usage:"
	@echo "  make clean      - Remove database files, db directory, and photo cache"
	@echo ""
	@echo "Testing:"
	@echo "  make test       - Run all tests"
	@echo "  make test-v     - Run all tests with verbose output"
	@echo "  make test-cov   - Run tests with coverage report"
	@echo "  make test-file FILE=<path>  - Run a specific test file"
	@echo "                    Example: make test-file FILE=tests/db/test_storage.py"
	@echo "  make test-case CASE=<path>  - Run a specific test case"
	@echo "                    Example: make test-case CASE=tests/db/test_storage.py::test_init_creates_tables"
	@echo "  make test-match MATCH=<pattern> - Run tests matching a keyword"
	@echo "                    Example: make test-match MATCH=storage"

# The cleanup command
clean:
	@echo "Cleaning up project files..."
	rm -rf $(DB_DIR)
	rm -rf $(CHROMA_DB_DIR)
	rm -rf $(CACHE_DIR)
	@echo "Cleanup complete."

ingest:
	@echo "Starting photo ingestion..."
	$(PYTHON) $(MAIN) ingest

cluster:
	@echo "Starting clustering..."
	$(PYTHON) $(MAIN) cluster

appv2:
	@echo "Starting App V2..."
	$(STREAMLIT) run appv2.py

# ============================================
# Testing Targets
# ============================================

# Run all tests
test:
	$(PYTEST) $(TEST_DIR)

# Run all tests with verbose output
test-v:
	$(PYTEST) $(TEST_DIR) -v

# Run tests with coverage report
test-cov:
	$(PYTEST) $(TEST_DIR) --cov=src --cov-report=term-missing

# Run a specific test file
# Usage: make test-file FILE=tests/db/test_storage.py
test-file:
ifndef FILE
	@echo "Error: FILE is not set. Usage: make test-file FILE=tests/db/test_storage.py"
	@exit 1
endif
	$(PYTEST) $(FILE) -v

# Run a specific test case
# Usage: make test-case CASE=tests/db/test_storage.py::test_init_creates_tables
test-case:
ifndef CASE
	@echo "Error: CASE is not set. Usage: make test-case CASE=tests/db/test_storage.py::test_init_creates_tables"
	@exit 1
endif
	$(PYTEST) $(CASE) -v

# Run tests matching a keyword pattern
# Usage: make test-match MATCH=storage
test-match:
ifndef MATCH
	@echo "Error: MATCH is not set. Usage: make test-match MATCH=storage"
	@exit 1
endif
	$(PYTEST) $(TEST_DIR) -v -k "$(MATCH)"