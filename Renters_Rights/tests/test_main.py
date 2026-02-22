from unittest.mock import patch, MagicMock
from main import main


@patch("main.gr.ChatInterface")
def test_main(mock_chat_interface):
    """Test that main initializes and launches the Gradio Interface."""
    mock_instance = MagicMock()
    mock_chat_interface.return_value = mock_instance

    main()

    mock_chat_interface.assert_called_once()
    mock_instance.launch.assert_called_once()
