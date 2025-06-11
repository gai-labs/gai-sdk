code --install-extension ms-toolsai.jupyter
code --install-extension charliermarsh.ruff
code --install-extension esbenp.prettier-vscode
code --install-extension bierner.markdown-mermaid
code --install-extension ms-azuretools.vscode-docker
code --install-extension ms-azuretools.vscode-containers
code --install-extension github.copilot
source ${UV_PROJECT_ENVIRONMENT}/bin/activate \
    && uv pip install -e ".[dev]"

# source ${UV_PROJECT_ENVIRONMENT}/bin/activate \
#     && uv pip install -e ".[dev]" \
#     && cd ${PROJECT_DIR}/mcp-servers/modelcontextprotocol/servers \
#     && npm i
   


