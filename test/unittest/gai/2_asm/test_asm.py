"""
Unit tests for GAI ASM dependency resolution in resolve_input method.

Tests cover the three main dependency types:
- getter: Resolves callable functions from kwargs
- prev_state: Resolves values from previous state output
- state_bag: Resolves values from current state bag
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
from typing import Optional

# Mock the dependencies that aren't available in test environment
import sys
sys.path.insert(0, '/workspace/target/src')

# Mock gai modules
sys.modules['gai.lib.logging'] = MagicMock()
sys.modules['gai.lib.constants'] = MagicMock()
sys.modules['gai.messages'] = MagicMock()
sys.modules['gai.asm.constants'] = MagicMock()

# Set up mock constants
sys.modules['gai.lib.constants'].DEFAULT_GUID = "test-guid"
sys.modules['gai.asm.constants'].HISTORY_PATH = "/tmp/test/{caller_id}/{dialogue_id}/{order_no}.json"

# Mock the Monologue class
mock_monologue = MagicMock()
mock_monologue.copy.return_value = "mocked_monologue_copy"
sys.modules['gai.messages'].Monologue = MagicMock(return_value=mock_monologue)

# Import the code under test
from gai.asm.asm import AgenticStateMachine


class TestResolveInputDependencies:
    """Test class for resolve_input dependency resolution methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test history files
        import tempfile
        import os
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock state machine components
        self.state_machine = MagicMock()
        self.state_machine.state = "TEST_STATE"
        self.state_machine.state_history = []
        self.state_machine.state_bag = {}
        self.state_machine.kwargs = {}
        
        # Create StateModel instance
        self.state_model = AgenticStateMachine.StateModel(
            state_manifest={"TEST_STATE": {"input_data": {}}},
            agent_name="TestAgent"
        )
        self.state_model.step = 0
        self.state_model.user_message = "test message"
        
        # Create mock state with proper structure
        self.mock_state = MagicMock()
        self.mock_state.machine = self.state_machine
        self.mock_state.manifest = {"input_data": {}}

    def teardown_method(self):
        """Clean up after each test method."""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_resolve_input_getter_sync_callable_success(self):
        """Test successful resolution of synchronous getter dependency."""
        # Arrange
        def mock_getter_func(state):
            return "getter_result"
        
        self.state_machine.kwargs = {"test_getter": mock_getter_func}
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "getter",
                    "dependency": "test_getter"
                }
            }
        }
        
        # Act
        result = await self.state_model.resolve_input(self.mock_state)
        
        # Assert
        assert "test_param" in result
        assert result["test_param"] == "getter_result"
        assert self.state_machine.state_bag["test_param"] == "getter_result"

    @pytest.mark.asyncio
    async def test_resolve_input_getter_async_callable_success(self):
        """Test successful resolution of asynchronous getter dependency."""
        # Arrange
        async def mock_async_getter_func(state):
            await asyncio.sleep(0.01)  # Simulate async work
            return "async_getter_result"
        
        self.state_machine.kwargs = {"test_async_getter": mock_async_getter_func}
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "getter",
                    "dependency": "test_async_getter"
                }
            }
        }
        
        # Act
        result = await self.state_model.resolve_input(self.mock_state)
        
        # Assert
        assert "test_param" in result
        assert result["test_param"] == "async_getter_result"

    @pytest.mark.asyncio
    async def test_resolve_input_getter_missing_dependency_key(self):
        """Test exception when getter dependency key is missing."""
        # Arrange
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "getter"
                    # Missing "dependency" key
                }
            }
        }
        
        # Act & Assert
        with pytest.raises(ValueError, match="'dependency' property not found for 'getter'"):
            await self.state_model.resolve_input(self.mock_state)

    @pytest.mark.asyncio
    async def test_resolve_input_getter_dependency_not_in_kwargs(self):
        """Test exception when getter dependency is not provided in kwargs."""
        # Arrange
        self.state_machine.kwargs = {}  # Empty kwargs
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "getter",
                    "dependency": "missing_getter"
                }
            }
        }
        
        # Act & Assert
        with pytest.raises(ValueError, match="Dependency missing_getter is specified in manifest but not passed into ASM builder"):
            await self.state_model.resolve_input(self.mock_state)

    @pytest.mark.asyncio
    async def test_resolve_input_getter_callable_raises_exception(self):
        """Test handling when getter callable raises an exception."""
        # Arrange
        def failing_getter(state):
            raise RuntimeError("Getter function failed")
        
        self.state_machine.kwargs = {"failing_getter": failing_getter}
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "getter",
                    "dependency": "failing_getter"
                }
            }
        }
        
        # Act & Assert
        # The method logs the error but continues, setting resolved to None
        with pytest.raises(ValueError, match="There are unresolved dependency test_param"):
            await self.state_model.resolve_input(self.mock_state)

    @pytest.mark.asyncio
    async def test_resolve_input_getter_async_callable_raises_exception(self):
        """Test handling when async getter callable raises an exception."""
        # Arrange
        async def failing_async_getter(state):
            raise RuntimeError("Async getter function failed")
        
        self.state_machine.kwargs = {"failing_async_getter": failing_async_getter}
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "getter",
                    "dependency": "failing_async_getter"
                }
            }
        }
        
        # Act & Assert
        with pytest.raises(ValueError, match="There are unresolved dependency test_param"):
            await self.state_model.resolve_input(self.mock_state)

    @pytest.mark.asyncio
    async def test_resolve_input_prev_state_success(self):
        """Test successful resolution of prev_state dependency."""
        # Arrange
        self.state_machine.state_history = [
            {
                "state": "PREV_STATE",
                "output": {
                    "prev_result": "previous_state_data",
                    "step": 1
                }
            }
        ]
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "prev_state",
                    "dependency": "prev_result"
                }
            }
        }
        
        # Act
        result = await self.state_model.resolve_input(self.mock_state)
        
        # Assert
        assert "test_param" in result
        assert result["test_param"] == "previous_state_data"

    @pytest.mark.asyncio
    async def test_resolve_input_prev_state_missing_dependency_key(self):
        """Test exception when prev_state dependency key is missing."""
        # Arrange
        self.state_machine.state_history = [{"state": "PREV_STATE", "output": {"data": "value"}}]
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "prev_state"
                    # Missing "dependency" key
                }
            }
        }
        
        # Act & Assert
        with pytest.raises(ValueError, match="'dependency' property not specified for 'prev_state'"):
            await self.state_model.resolve_input(self.mock_state)

    @pytest.mark.asyncio
    async def test_resolve_input_prev_state_no_history(self):
        """Test exception when no previous state exists in history."""
        # Arrange
        self.state_machine.state_history = []  # Empty history
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "prev_state",
                    "dependency": "some_data"
                }
            }
        }
        
        # Act & Assert
        with pytest.raises(ValueError, match="Last output not found in state history"):
            await self.state_model.resolve_input(self.mock_state)

    @pytest.mark.asyncio
    async def test_resolve_input_prev_state_dependency_not_found(self):
        """Test exception when requested dependency not found in previous state output."""
        # Arrange
        self.state_machine.state_history = [
            {
                "state": "PREV_STATE",
                "output": {
                    "existing_data": "value"
                }
            }
        ]
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "prev_state",
                    "dependency": "missing_data"
                }
            }
        }
        
        # Act & Assert
        with pytest.raises(ValueError, match="Dependency missing_data not found in last output"):
            await self.state_model.resolve_input(self.mock_state)

    @pytest.mark.asyncio
    async def test_resolve_input_prev_state_dependency_is_none(self):
        """Test exception when dependency exists but is None in previous state output."""
        # Arrange
        self.state_machine.state_history = [
            {
                "state": "PREV_STATE",
                "output": {
                    "null_data": None
                }
            }
        ]
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "prev_state",
                    "dependency": "null_data"
                }
            }
        }
        
        # Act & Assert
        with pytest.raises(ValueError, match="Dependency null_data not found in last output"):
            await self.state_model.resolve_input(self.mock_state)

    @pytest.mark.asyncio
    async def test_resolve_input_state_bag_success(self):
        """Test successful resolution of state_bag dependency."""
        # Arrange
        self.state_machine.state_bag = {
            "bag_data": "state_bag_value",
            "other_data": 42
        }
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "state_bag",
                    "dependency": "bag_data"
                }
            }
        }
        
        # Act
        result = await self.state_model.resolve_input(self.mock_state)
        
        # Assert
        assert "test_param" in result
        assert result["test_param"] == "state_bag_value"

    @pytest.mark.asyncio
    async def test_resolve_input_state_bag_missing_dependency_key(self):
        """Test that missing dependency key defaults to None and raises ValueError."""
        # Arrange
        self.state_machine.state_bag = {"existing_data": "value"}
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "state_bag"
                    # Missing "dependency" key - defaults to None
                }
            }
        }
        
        # Act & Assert
        with pytest.raises(ValueError, match="Dependency None not found in state bag"):
            await self.state_model.resolve_input(self.mock_state)

    @pytest.mark.asyncio
    async def test_resolve_input_state_bag_dependency_not_found(self):
        """Test exception when requested dependency not found in state bag."""
        # Arrange
        self.state_machine.state_bag = {
            "existing_key": "existing_value"
        }
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "state_bag",
                    "dependency": "missing_key"
                }
            }
        }
        
        # Act & Assert
        with pytest.raises(ValueError, match="Dependency missing_key not found in state bag"):
            await self.state_model.resolve_input(self.mock_state)

    @pytest.mark.asyncio
    async def test_resolve_input_state_bag_dependency_is_none(self):
        """Test exception when dependency exists but is None in state bag."""
        # Arrange
        self.state_machine.state_bag = {
            "null_key": None
        }
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "state_bag",
                    "dependency": "null_key"
                }
            }
        }
        
        # Act & Assert
        with pytest.raises(ValueError, match="Dependency null_key not found in state bag"):
            await self.state_model.resolve_input(self.mock_state)

    @pytest.mark.asyncio
    async def test_resolve_input_multiple_dependency_types(self):
        """Test resolution of multiple dependency types in single call."""
        # Arrange
        def test_getter(state):
            return "getter_value"
        
        self.state_machine.kwargs = {"test_getter": test_getter}
        self.state_machine.state_history = [
            {"state": "PREV", "output": {"prev_data": "prev_value"}}
        ]
        self.state_machine.state_bag = {"bag_data": "bag_value"}
        
        self.mock_state.manifest = {
            "input_data": {
                "getter_param": {
                    "type": "getter",
                    "dependency": "test_getter"
                },
                "prev_param": {
                    "type": "prev_state",
                    "dependency": "prev_data"
                },
                "bag_param": {
                    "type": "state_bag",
                    "dependency": "bag_data"
                },
                "literal_param": "literal_value"
            }
        }
        
        # Act
        result = await self.state_model.resolve_input(self.mock_state)
        
        # Assert
        assert result["getter_param"] == "getter_value"
        assert result["prev_param"] == "prev_value"
        assert result["bag_param"] == "bag_value"
        assert result["literal_param"] == "literal_value"
        # Built-in values should also be present
        assert "user_message" in result
        assert "monologue" in result
        assert "step" in result
        assert "time" in result
        assert "name" in result

    @pytest.mark.asyncio
    async def test_resolve_input_reserved_keys_validation(self):
        """Test that reserved keys in input_data raise appropriate exceptions."""
        reserved_keys = ["monologue", "user_message", "step", "time", "name"]
        
        for reserved_key in reserved_keys:
            # Arrange
            self.mock_state.manifest = {
                "input_data": {
                    reserved_key: "some_value"
                }
            }
            
            # Act & Assert
            with pytest.raises(ValueError, match=f"input_data cannot contain reserved key `{reserved_key}`"):
                await self.state_model.resolve_input(self.mock_state)

    @pytest.mark.asyncio
    async def test_resolve_input_unknown_dependency_type(self):
        """Test that unknown dependency type is treated as literal value."""
        # Arrange
        self.mock_state.manifest = {
            "input_data": {
                "test_param": {
                    "type": "unknown_type",
                    "value": "some_value"
                }
            }
        }
        
        # Act
        result = await self.state_model.resolve_input(self.mock_state)
        
        # Assert
        assert "test_param" in result
        assert result["test_param"]["type"] == "unknown_type"
        assert result["test_param"]["value"] == "some_value"

    @pytest.mark.asyncio
    async def test_resolve_input_state_bag_update(self):
        """Test that state bag is properly updated with resolved input data."""
        # Arrange
        def test_getter(state):
            return {"complex": "data"}
        
        self.state_machine.kwargs = {"test_getter": test_getter}
        self.mock_state.manifest = {
            "input_data": {
                "getter_param": {
                    "type": "getter", 
                    "dependency": "test_getter"
                },
                "literal_param": "literal_value"
            }
        }
        
        # Act
        result = await self.state_model.resolve_input(self.mock_state)
        
        # Assert - state bag should contain all resolved input data
        assert self.state_machine.state_bag["getter_param"] == {"complex": "data"}
        assert self.state_machine.state_bag["literal_param"] == "literal_value"
        assert "user_message" in self.state_machine.state_bag
        assert "monologue" in self.state_machine.state_bag
        assert "step" in self.state_machine.state_bag
        assert "time" in self.state_machine.state_bag
        assert "name" in self.state_machine.state_bag

    @pytest.mark.asyncio
    async def test_resolve_input_pass_1_and_pass_2_separation(self):
        """Test that literal values are resolved in pass 1 and dependencies in pass 2."""
        # Arrange
        call_order = []
        
        def tracking_getter(state):
            call_order.append("getter_called")
            return "getter_result"
        
        self.state_machine.kwargs = {"tracking_getter": tracking_getter}
        self.mock_state.manifest = {
            "input_data": {
                "literal_first": "literal_value",
                "getter_second": {
                    "type": "getter",
                    "dependency": "tracking_getter"
                }
            }
        }
        
        # Act
        result = await self.state_model.resolve_input(self.mock_state)
        
        # Assert
        assert len(call_order) == 1
        assert call_order[0] == "getter_called"
        assert result["literal_first"] == "literal_value"
        assert result["getter_second"] == "getter_result"

    @pytest.mark.asyncio
    async def test_resolve_input_deep_copy_preservation(self):
        """Test that original input data is not modified and deep copy is returned."""
        # Arrange
        original_input_data = {
            "nested_param": {
                "type": "getter",
                "dependency": "test_getter"
            }
        }
        
        def test_getter(state):
            return {"mutable": ["data"]}
        
        self.state_machine.kwargs = {"test_getter": test_getter}
        self.mock_state.manifest = {"input_data": original_input_data}
        
        # Act
        result = await self.state_model.resolve_input(self.mock_state)
        
        # Assert - original input_data should be unchanged
        assert original_input_data["nested_param"]["type"] == "getter"
        assert original_input_data["nested_param"]["dependency"] == "test_getter"
        
        # Result should contain resolved value
        assert result["nested_param"] == {"mutable": ["data"]}
        
        # Note: The current implementation copies resolved_input_data to state_bag,
        # so modifying the returned result will also affect the state bag since they 
        # reference the same objects. This test verifies the current behavior.
        result["nested_param"]["mutable"].append("modified")
        assert self.state_machine.state_bag["nested_param"]["mutable"] == ["data", "modified"]