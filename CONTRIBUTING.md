# Contributing to NeuroPHorm

First off, thank you for taking the time to contribute! NeuroPHorm thrives on community feedback, new ideas, and rigorous quality checks. This document outlines how to set up your environment, propose changes, and submit a pull request.

## 🛠️ Development setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mohimozaffari/neurophorm.git
   cd neurophorm
   ```
2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
4. **Install optional tooling** (linting, formatting)
   ```bash
   pip install black flake8 isort pytest
   ```

> 💡 Tip: Pin additional dependencies (e.g., `pre-commit`, `mypy`) in your own workflow, but keep `requirements.txt` focused on runtime needs.

## 🧪 Testing & quality checks

Before submitting a pull request, run the relevant checks locally:

```bash
pytest
flake8 neurophorm
black --check neurophorm examples
isort --check-only neurophorm examples
```

If you are contributing a feature, add targeted unit tests or notebooks to demonstrate usage. For bug fixes, regression tests are highly encouraged.

## 🧭 Issue workflow

- **Bug reports** – Include a minimal reproducible example, expected vs. actual behaviour, and environment details.
- **Feature requests** – Outline the problem you are solving, proposed API, and any alternative approaches.
- **Questions** – Use GitHub Discussions or issues with the `question` label.

Please search existing issues before opening a new one.

## 🧱 Coding guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) and write clear, type-hinted functions.
- Prefer explicit imports over wildcard imports.
- Keep functions focused and reusable; break down long workflows into composable helpers.
- Document non-trivial logic with docstrings and in-line comments when necessary.
- Maintain backwards compatibility when feasible. Call out breaking changes early in the discussion.

## 📝 Commit & PR conventions

- Write descriptive commit messages in the present tense (e.g., `Add helper for batch persistence export`).
- Reference GitHub issues using `Fixes #123` or `Closes #123` when applicable.
- Keep pull requests focused. If you plan a large refactor, discuss it in an issue before implementation.
- Update documentation (README, docstrings, CHANGELOG) alongside code changes when behaviour or APIs change.
- Provide screenshots or GIFs for visual changes.

## 🙌 Code of Conduct

Please review the [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## 💌 Need help?

If you get stuck, open a draft pull request with your questions or reach out via the issue tracker. We are happy to collaborate!

Thank you for helping make NeuroPHorm better.
