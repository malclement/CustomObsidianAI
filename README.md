# python_template

## Pre-commit

1. Install pre-commit :
   ```bash
   pip install pre-commit
   ```
2. Run :
   ```bash
   pre-commit instal
   ```

#### Note

You can skip the pre-commit validation using `-n`:

```bash
git commit -m 'my_message' -n
```

---

## Virtual Environment

1. Instal virtualenv :
   ```bash
   pip install virtualenv
   ```
2. Create a virutal environment :
   Locate yourself at the root of your project
   ```bash
    python<version> -m venv env
   ```
3. Activate :
   ```bash
   source env/bin/activate
   ```
