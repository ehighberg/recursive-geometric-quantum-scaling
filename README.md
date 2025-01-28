# Recursive Geometric Quantum Scaling (RGQS)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ehighberg/recursive-geometric-quantum-scaling.git
cd recursive-geometric-quantum-scaling
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Using the Streamlit Interface

1. Launch the Streamlit app:
```bash
streamlit run --server.port 3001 app.py
```
Specify a different port if needed.

## Testing

Run the test suite:

```bash
pytest tests/
```

For coverage report:

```bash
pytest --cov=simulations tests/
```
