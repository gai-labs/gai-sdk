import os
import io
import shutil
import base64
from PIL import Image
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp-common")
available_directories = []
tools_instance = None


@mcp.tool()
def delete_file(file_path: str) -> bool:
    """
    Delete a file if it exists.

    Args:
        file_path (str): Path to the file to delete.

    Returns:
        bool: True if the file was deleted, False if it did not exist.
    """
    file_path = os.path.expanduser(file_path)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


@mcp.tool()
def move_file(src_path: str, dst_path: str) -> str:
    """
    Move a file from source to destination, handling cross-filesystem moves.

    Args:
        src_path (str): Source file path
        dst_path (str): Destination file path (full path including filename)

    Returns:
        str: Final destination path
    """
    # Expand user directory (~) for both paths
    src_path = os.path.expanduser(src_path)
    dst_path = os.path.expanduser(dst_path)

    # Create destination directory if it doesn't exist
    dst_dir = os.path.dirname(dst_path)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)

    try:
        # Try regular move first (faster if same filesystem)
        shutil.move(src_path, dst_path)
    except OSError as e:
        if e.errno == 18:  # EXDEV: cross-device link not permitted
            # Copy then delete for cross-filesystem moves
            shutil.copy2(src_path, dst_path)
            os.unlink(src_path)
        else:
            raise e

    return dst_path


@mcp.tool()
def is_exist(file_path: str):
    """
    Simple check to verify if a directory exists.
    Args:
        name: Name of the directory to check.
        directory: Path in which to check for the directory.
    Returns:
        bool: True if the directory exists, False otherwise.
    """
    return os.path.exists(file_path)


@mcp.tool()
def read_text_file(file_path: str):
    """
    Read the contents of a text file.
    Args:
        file_path (str): Path to the text file to read.
    Returns:
        dict: Result containing the file contents or an error message if the file does not exist.
    """

    try:
        file_path = os.path.expanduser(file_path)
        if not os.path.exists(file_path):
            return {"error": f"File {file_path} does not exist."}
        with open(file_path, "r") as file:
            content = file.read()

        return {"content": content}

    except Exception as e:
        return {"error": f"Failed to read file {file_path}: {str(e)}"}


@mcp.tool()
def read_image_file(file_path: str):
    """
    Read the contents of an image file into a base64-encoded string.
    Resizes image to fit within 512x512 pixels while maintaining aspect ratio if needed.
    Args:
        file_path (str): Path to the image file to read.
    Returns:
        dict: Result containing the base64-encoded content of the image or an error message if the file does not exist.
    """
    try:
        file_path = os.path.expanduser(file_path)
        if not os.path.exists(file_path):
            return {"error": f"Image file {file_path} does not exist."}

        # Open and process the image
        with Image.open(file_path) as img:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Check if resizing is needed
            width, height = img.size
            max_dimension = max(width, height)

            if max_dimension > 512:
                # Calculate new dimensions maintaining aspect ratio
                if width > height:
                    new_width = 512
                    new_height = int((height * 512) / width)
                else:
                    new_height = 512
                    new_width = int((width * 512) / height)

                # Resize the image
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(
                    f"Resized image from {width}x{height} to {new_width}x{new_height}"
                )

            # Determine the format and MIME type
            original_format = img.format or "JPEG"
            if file_path.lower().endswith((".jpg", ".jpeg")):
                save_format = "JPEG"
                mime_type = "image/jpeg"
            elif file_path.lower().endswith(".png"):
                save_format = "PNG"
                mime_type = "image/png"
            elif file_path.lower().endswith(".webp"):
                save_format = "WEBP"
                mime_type = "image/webp"
            else:
                # Default to JPEG for other formats
                save_format = "JPEG"
                mime_type = "image/jpeg"

            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format=save_format, quality=85, optimize=True)
            buffer.seek(0)

            base64_content = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return {
                "content": {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "mime_type": mime_type,
                        "data": base64_content,
                    },
                }
            }

    except ImportError:
        return {
            "error": "PIL (Pillow) library is required for image processing. Install with: pip install Pillow"
        }
    except Exception as e:
        return {"error": f"Failed to read image file {file_path}: {str(e)}"}


@mcp.tool()
def find_files_containing_string(directory: str, search_string: str):
    """
    Find files in a directory that contain a specific string.
    Args:
        directory (str): Directory to search in.
        search_string (str): String to search for in files.
    Returns:
        list: List of file paths containing the search string.
    """
    matching_files = []
    directory = os.path.expanduser(directory)

    if not os.path.exists(directory):
        return {"error": f"Directory {directory} does not exist."}

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    if search_string in f.read():
                        matching_files.append(file_path)
            except (IOError, UnicodeDecodeError):
                # Skip files that can't be read or decoded
                continue

    return {"files": matching_files}


@mcp.tool()
def file_tree(file_path: str, max_depth: int = -1) -> dict:
    """
    Generate a visual tree of files and directories starting from file_path.

    Args:
        file_path (str): Root directory path to generate the tree from.
        max_depth (int): Maximum depth to recurse into subdirectories. -1 means no limit.

    Returns:
        dict: Dictionary with a "tree" key containing the tree as a string.
    """
    import os

    def generate_tree(path: str, prefix: str = "", depth: int = 0) -> str:
        if max_depth >= 0 and depth > max_depth:
            return ""
        tree_lines = []
        try:
            entries = sorted(os.listdir(path))
        except Exception:
            return f"{prefix}└── [Permission Denied]"

        for i, entry in enumerate(entries):
            full_path = os.path.join(path, entry)
            connector = "└── " if i == len(entries) - 1 else "├── "
            tree_lines.append(f"{prefix}{connector}{entry}")
            if os.path.isdir(full_path):
                extension = "    " if i == len(entries) - 1 else "│   "
                tree_lines.append(
                    generate_tree(full_path, prefix + extension, depth + 1)
                )
        return "\n".join(tree_lines)

    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        return {"error": f"Path {file_path} does not exist."}
    if not os.path.isdir(file_path):
        return {"error": f"Path {file_path} is not a directory."}

    tree_output = f"{os.path.basename(file_path) or file_path}\n"
    tree_output += generate_tree(file_path)
    return {"tree": tree_output}


@mcp.tool()
def run_pytest(test_file_path: str):
    """
    Run pytest on a specified test file and return the results.
    Args:
        test_file_path (str): Path to the test file to run.
    Returns:
        str: Result of the pytest run, including any errors or output.
    """
    import subprocess

    try:
        test_file_path = os.path.expanduser(test_file_path)
        if not os.path.exists(test_file_path):
            return {"error": f"Test file {test_file_path} does not exist."}

        # Run pytest command
        result = subprocess.run(
            ["pytest", test_file_path],
            capture_output=True,
            text=True,
            check=False,
        )

        # Return the output and error messages
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except Exception as e:
        return {"error": f"Failed to run pytest on {test_file_path}: {str(e)}"}


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
