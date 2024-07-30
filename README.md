# GAI SDK

## Installation

### 1. Create conda env

```bash
conda create -n ttt python=3.10.10 -y
conda activate ttt
```

### 2. Install Packages

```bash
pip install -e ".[dev]"
```

### 3. Initialise

```bash
gai init
```

### 4. Download model

```bash
gai pull exllamav2-mistral7b
```

### 5. Start server

a) Click on Run and Debug icon in Activity Bar -> Python Debugger: TTT Api Server -> Start Debugging

b) Note that it may fail because the Debug Console has not activated the conda environment. Just stop the debugger and go to the DEBUG CONSOLE. Activate the conda environment `conda activate ttt` and press F5.

If it is successfully started, you will see the following:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:12031 (Press CTRL+C to quit)
```

## Quick Start

Run [Quick Start Guide](./doc/1-quickstart.ipynb)
