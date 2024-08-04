# GAI SDK

## A. Install from Github

### 1. Create conda env

```bash
conda create -n ttt python=3.10.10 -y
conda activate ttt
```

### 2. Clone Repository

```bash
git clone https://github.com/kakkoii1337/gai-sdk
```

### 3. Install Packages

```bash
cd gai-sdk
pip install -e ".[dev]"
```

### 4. Initialise

```bash
gai init
```

### 5. Download model

```bash
gai pull exllamav2-mistral7b
```

## B. Install from Pypi

### 1. Create conda env

```bash
conda create -n ttt python=3.10.10 -y
conda activate ttt
```

### 2. Install Packages

```bash
pip install gai-sdk[ttt]
```

### 4. Initialise

```bash
gai init
```

### 5. Download model

```bash
gai pull exllamav2-mistral7b
```

## Start server

```
gai docker start
```

If it is successfully started, you will see the following:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:12031 (Press CTRL+C to quit)
```

## Quick Start

Run [Quick Start Guide](./doc/1-quickstart.ipynb)
