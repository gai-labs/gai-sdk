import os
import pytest
import tomli

@pytest.fixture
def pyproject_data():
    """Load and return the pyproject.toml data for testing."""
    # Find the pyproject.toml file by navigating up from the current test directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pyproject_path = os.path.join(base_dir, "pyproject.toml")
    
    # Check if the file exists
    if not os.path.exists(pyproject_path):
        pytest.fail(f"pyproject.toml not found at {pyproject_path}")
    
    # Load the TOML file
    with open(pyproject_path, "rb") as f:
        return tomli.load(f)

def test_package_metadata(pyproject_data):
    """Test that package metadata is correctly configured."""
    project = pyproject_data["project"]
    
    assert "name" in project, "Project name should be specified"
    assert "version" in project, "Project version should be specified"
    assert "description" in project, "Project description should be specified"
    assert "readme" in project, "Project readme should be specified"
    assert "requires-python" in project, "Python version requirement should be specified"
    assert ">=" in project["requires-python"], "Python version requirement should specify minimum version"

def test_dependencies(pyproject_data):
    """Test that dependencies are correctly specified."""
    dependencies = pyproject_data["project"]["dependencies"]
    
    assert isinstance(dependencies, list), "Dependencies should be a list"
    assert len(dependencies) > 0, "Should have at least one dependency"
    
    # Check for common dependencies
    dependency_names = " ".join(dependencies).lower()
    assert "openai>=1.43.0,<2.0.0" in dependency_names, "Should include HTTP client library"

def test_classifiers(pyproject_data):
    """Test that project classifiers are correctly specified."""
    classifiers = pyproject_data["project"].get("classifiers", [])
    
    assert len(classifiers) > 0, "Should have at least one classifier"
    assert any("Python" in c for c in classifiers), "Should specify Python version classifiers"
    assert any("License" in c for c in classifiers), "Should specify license classifier"
    assert any("Development Status" in c for c in classifiers), "Should specify development status"

def test_project_urls(pyproject_data):
    """Test that project URLs are correctly specified."""
    urls = pyproject_data["project"]["urls"]
    
    assert "documentation" in urls, "Documentation URL should be specified"
    assert "https://gai-labs.github.io/gai/docs" in urls["documentation"], "Documentation URL should be correct"

def test_project_authors(pyproject_data):
    """Test that project authors are correctly specified."""
    authors = pyproject_data["project"]["authors"]
    
    assert len(authors) > 0, "Should have at least one author"
    author = authors[0]
    assert "name" in author, "Author should have a name"
    assert "email" in author, "Author should have an email"
    assert "@" in author["email"], "Email should be valid"
