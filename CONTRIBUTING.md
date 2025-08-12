# Contributing Guide

Thank you for considering a contribution! This project is developed in milestones with a strong emphasis on Test-Driven Development (TDD), clarity, and maintainability.

## Principles
- Prefer clear, readable code with type hints and docstrings.
- Add tests for every feature/fix; keep coverage high.
- Follow PEP 8 and project tooling (ruff, black, mypy).
- Avoid unnecessary dependencies; pin versions where appropriate.

## Workflow
1. Fork and create a branch.
2. Write a failing test in `tests/` that expresses the change.
3. Implement the change with minimal code.
4. Ensure all checks pass locally:
   - `ruff --fix .`
   - `black .`
   - `mypy .`
   - `pytest -q`
5. Open a PR describing the change and linking any related issues.

## Commit messages
Use concise, imperative messages (e.g., "Add RAGAS evaluation harness").

## Security
Do not include secrets in code or logs. Use `.env` for local secrets and never commit it.


