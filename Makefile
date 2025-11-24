#################################################################################
## Supabase Commands
#################################################################################

supabase-create: ## Create Supabase database
	@echo "Creating Supabase database..."
	uv run python src/infrastructure/supabase/create_db.py