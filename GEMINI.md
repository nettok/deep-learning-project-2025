# Gemini Development Guidelines

This document outlines the coding standards and practices for this project.

## Python Best Practices

### 1. Type Hinting
Use type annotations to improve code readability and IDE support. While Python is dynamically typed, hints help document expectations for function arguments and return values.

*   **When to use:**
    *   Function signatures (arguments and return types).
    *   Class attributes.
    *   Complex data structures (using `typing.List`, `typing.Dict`, `typing.Optional`, etc., or standard collections in Python 3.9+).
*   **Pandas/TensorFlow:** Use `pd.DataFrame`, `np.ndarray`, and `tf.Tensor` to indicate data types.

**Example:**
```python
import pandas as pd
import tensorflow as tf
from typing import Optional

def preprocess_data(df: pd.DataFrame, target_col: str) -> tuple[tf.Tensor, tf.Tensor]:
    """Separates features and target."""
    # ... implementation ...
    return features, target
```

### 2. Functional & Immutable Style
Prefer a functional programming style where it enhances clarity and reduces side effects.

*   **Immutability:** Avoid mutating data structures in place when possible. Return new objects instead of modifying arguments.
    *   *Prefer:* `new_df = df.drop(columns=['id'])`
    *   *Avoid:* `df.drop(columns=['id'], inplace=True)`
*   **Pure Functions:** Write functions that rely only on their inputs and produce consistent outputs without changing global state.
*   **Comprehensions:** Use list/dict comprehensions for concise data transformation, but avoid them if logic becomes too complex (readability first).

**Example:**
```python
# Good: Returns a new list, doesn't modify original
def normalize_names(names: list[str]) -> list[str]:
    return [name.strip().lower() for name in names]

# Avoid: Modifies list in-place (unless performance dictates otherwise)
def normalize_names_inplace(names: list[str]) -> None:
    for i in range(len(names)):
        names[i] = names[i].strip().lower()
```

### 3. Readability & Pythonic Code
*   Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
*   Write clear, descriptive variable and function names.
*   Keep functions small and focused on a single task.

### 4. Fail Fast
Functions should raise exceptions early and clearly when an error occurs, rather than returning `None` or an empty value. This prevents silent failures and makes debugging easier.

*   **Prefer:** Raising specific exceptions (`FileNotFoundError`, `ValueError`, etc.) when preconditions are not met or errors occur.
*   **Avoid:** Returning `None` or an empty data structure to indicate an error, as this can lead to downstream `AttributeError` or unexpected behavior.

**Example:**
```python
# Good: Raises FileNotFoundError if file doesn't exist
def load_config(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)

# Avoid: Returns None on error, requiring calling code to check
def load_config_lenient(file_path: str) -> Optional[dict]:
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
```

### 5. Virtual Environment
To run this project, the `.venv` python binary needs to be used.

### 6. Git Operations
The user will handle all git operations (adding, committing, etc.) themselves. Only do it if specifically asked to.
