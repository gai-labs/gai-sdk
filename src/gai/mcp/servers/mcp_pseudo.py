from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp-pseudo")
available_directories = []
tools_instance = None


@mcp.tool()
def user_input() -> str:
    """
    When you see this tool, it means that you may interrupt the conversation to ask the user for input.
    When calling this tool, you need to be specific and clear about what you need from the user.
    You need to ask in such a way that you can get a specific answer after this tool is called.
    This is a pseudo tool and so do not inform the user you are using it. You just have to use it.
    ALWAYS REMEMBER. If you need user input, you MUST call this tool.
    Args:
        N.A.
    Returns:
        str: Input from the user
    """
    return "User input is required."


# @mcp.tool()
# def task_completed() -> str:
#     """
#     Call this function to trigger the completion of a task when user's goal is met and no further tool_use is required.
#     Args:
#         N.A.
#     Returns:
#         N.A.
#     """
#     raise Exception(
#         "This is a pseudo tool and serves only as a marker. It should not be called directly."
#     )


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
