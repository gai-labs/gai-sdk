from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp-pseudo")
available_directories = []
tools_instance = None


@mcp.tool()
def user_input() -> str:
    """
    Call this function if an input is expected from the user.
    Args:
        N.A.
    Returns:
        str: Input from the user
    """
    raise Exception(
        "This is a pseudo tool and serves only as a marker. It should not be called directly."
    )


@mcp.tool()
def task_completed() -> str:
    """
    Call this function to trigger the completion of a task when user's goal is met and no further tool_use is required.
    Args:
        N.A.
    Returns:
        N.A.
    """
    raise Exception(
        "This is a pseudo tool and serves only as a marker. It should not be called directly."
    )


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
