## Setup Instructions

### 1. Clone the Repository & navigate

```bash
git clone https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection
cd DCE-DSC-MRI_CodeCollection
```

### 2. Create a Virtual Environment (Optional but Recommended)
- Note: this project uses python version 3.8, if you already have this version then please proceed, else please checkout [setup virtual environment python3.8](/guide/setup_venv.md)

- Proceed only if the virtual environment for python 3.8 is setup correctly

### 3. Install Core Dependencies & requirements

- install the requirements
```bash
pip install -r requirements.txt
```
- install the setup.py
```bash
pip install .
```

### 4. Run Tests

```bash
pytest .
```
### 5. Execute the Notebooks ( if it is not already executed )

- Note: Before starting the documentation site, you must execute the notebooks.
- please checkout [render notebooks](/guide/render_notebooks.md)

### 5. Serve Locally

```bash
mkdocs serve
```