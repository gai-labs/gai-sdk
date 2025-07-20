# gai-init

gai-init is a utility for initializing Gai config and downloading local models.

## For Users

### a) To install Gai config

```bash
pip install gai-init
```

### b) To initialize Gai config

```bash
gai-init init
```

### c) To delete existing Gai config and start over

```bash
gai-init init --force
```

### d) To download local models

```bash
gai-init pull llama3.1-exl2
```

---

## For Contributors

### a) To publish a new version

Make sure .pypirc is configured with your PyPI credentials.

```bash
make publish
```
