# AI Agent Project

This project implements an AI agent using Python.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
- Copy `.env.example` to `.env`
- Update the values in `.env` with your configuration

## Development

- Use `black` for code formatting
- Use `isort` for import sorting
- Use `flake8` for linting
- Use `pytest` for testing

## Project Structure

```
.
├── .env                 # Environment variables
├── .gitignore          # Git ignore file
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## License

MIT 