## Execute the Notebooks


### 1. Execute All Notebooks in `notebooks/`

To execute all notebooks in the `notebooks/` folder (excluding the `old_nb`), use the following shell script:

```bash
find notebooks -type f -name "*.ipynb" ! -path "*/old_nb/*" | while read nb; do
    echo "Executing $nb"
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

> 💡 Requires `bash`. Run in WSL, or macOS/Linux terminal.

---
### 2. Execute a Specific Notebook (optional)

To execute a notebook and update its output cells **in place**:

```bash
jupyter nbconvert --to notebook --execute --inplace path/to/your_notebook.ipynb
```

### Example:

```bash
jupyter nbconvert --to notebook --execute --inplace docs/ToftsModel.ipynb
```