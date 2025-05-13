from .nodes import LoadDiTModel, LoadVAEModel, LoadTextEncoderModel, LoadGameImage, LoadMouseIcon, GameVideoGenerator, MatrixGameOutput

NODE_CLASS_MAPPINGS = {
    "LoadDiTModel": LoadDiTModel,
    "LoadVAEModel": LoadVAEModel,
    "LoadTextEncoderModel": LoadTextEncoderModel,
    "LoadGameImage": LoadGameImage,
    "LoadMouseIcon": LoadMouseIcon,
    "GameVideoGenerator": GameVideoGenerator,
    "MatrixGameOutput": MatrixGameOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDiTModel": "Load DiT Model",
    "LoadVAEModel": "Load VAE Model",
    "LoadTextEncoderModel": "Load TextEncoder Model",
    "LoadGameImage": "Load Game Image",
    "LoadMouseIcon": "Load Mouse Icon",
    "GameVideoGenerator": "Game Video Generator",
    "MatrixGameOutput": "MatrixGame Output",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
