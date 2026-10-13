# Contributing to Aptivra

Welcome! We welcome contributions to **Aptivra**, an open-source autonomous continual learning framework.

---

## Before you write code

1. Ensure your changes follow clean Python practices and type safety (`mypy --strict`).
2. Keep core architectural components modular and well-tested (`pytest`).
3. Ensure no hardcoded secrets or credentials are included.

## Making a change

- Branch from `main` or create a feature branch.
- Keep the change focused to a logical unit of work.
- Include unit tests for new features or bug fixes.
- Commit messages should clearly describe *why* and *what* was changed.

## Quality gates (MANDATORY)

Before submitting a PR, make sure all quality checks pass:

```bash
# Run pytest test suite
pytest

# Strict mypy type checking
mypy --strict aptivra

# Code style linting
ruff check .
```

## Opening a pull request

- Describe what changed and why.
- Provide test evidence or execution output demonstrating the fix/feature.
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting a problem

- **Security vulnerability** → Private security disclosure.
- **Defect or feature request** → Open a GitHub issue.

## License of contributions

All contributions are submitted under the terms of the [Apache License 2.0](LICENSE).
