# Contributing to markrel 🐟

Thank you for your interest in contributing to markrel! We welcome contributions from the community.

## 🐟 How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and markrel version

### Suggesting Features

We'd love to hear your ideas! Open an issue with:
- Feature description
- Use case
- Proposed API (if applicable)

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 🧪 Development Setup

```bash
git clone https://github.com/yourusername/markrel.git
cd markrel
pip install -e ".[dev]"
pytest tests/ -v
```

## 📝 Code Style

- Follow PEP 8
- Add type hints to new functions
- Write docstrings in Google style
- Keep functions focused and small

## 🧪 Testing

All contributions should include tests:

```python
def test_new_feature():
    model = MarkovRelevanceModel()
    # Your test here
    assert result == expected
```

Run tests before submitting:
```bash
pytest tests/ -v --cov=markrel
```

## 🙏 Thanks!

Every contribution helps make markrel better for everyone. Thank you for being part of the school! 🐟🐟🐟
