# Define the directories and files to remove
DB_DIR = .db
CHROMA_DB_DIR = db
CACHE_DIR = photos/cache
PYTHON = python3
MAIN = main.py
STREAMLIT = streamlit

.PHONY: clean help

# The default action if you just type 'make'
help:
	@echo "Usage:"
	@echo "  make clean    - Remove database files, db directory, and photo cache"

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

appv2:
	@echo "Starting App V2..."
	$(STREAMLIT) run appv2.py