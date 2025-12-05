#################################################################################
## Supabase Commands
#################################################################################

supabase-create: ## Create Supabase database
	@echo "Creating Supabase database..."
	uv run python -m src.infrastructure.supabase.create_db


qdrant-create-collection: ## Create Qdrant collection
	@echo "Creating Qdrant collection..."
	uv run python -m src.infrastructure.qdrant.create_collection