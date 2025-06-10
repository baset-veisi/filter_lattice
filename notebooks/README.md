# Filter Lattice Development Notebooks

This directory contains Jupyter notebooks for interactive development and testing of the filter lattice library.

## Setup

1. Install the required dependencies:
```bash
pip install -r ../requirements.txt
```

2. Start Jupyter Notebook:
```bash
jupyter notebook
```

## Notebooks

- `development.ipynb`: Main development notebook for testing and developing features
  - Contains utility functions for visualization
  - Examples of FIR and IIR filter testing
  - Interactive visualization of impulse and frequency responses
  - Easy to modify and extend for new features

## Development Workflow

1. Start Jupyter Notebook server
2. Open `development.ipynb`
3. Make changes to the library code in your editor
4. Use the notebook to test changes interactively
5. The notebook will automatically reload the library when you run cells

## Tips

- Use `%reload_ext autoreload` and `%autoreload 2` at the start of your notebook to automatically reload modules when they change
- Keep the notebook running while you make changes to the library code
- Use the notebook for quick prototyping and visualization
- When a feature is working, move the code to the main library 