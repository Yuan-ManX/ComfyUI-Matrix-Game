from .nodes import PlayGame

NODE_CLASS_MAPPINGS = {
    "PlayGame": PlayGame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlayGame": "Play Game",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
