bl_info = {
    "name": "SkinTokens - Mesh Auto-Rigging",
    "author": "Aero",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > SkinTokens",
    "description": "skeletal rigging and skinning weight prediction in Blender",
    "warning": "",
    "doc_url": "https://github.com/VAST-AI-Research/SkinTokens",
    "category": "Rigging",
}

from . import preferences
from . import operators
from . import ui


def register():
    preferences.register()
    operators.register()
    ui.register()


def unregister():
    ui.unregister()
    operators.unregister()
    preferences.unregister()


if __name__ == "__main__":
    register()
