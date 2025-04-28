from .nodes.video_captioning import VideoCaptioningNode
from .nodes.caption_refinement import CaptionRefinementNode

# Configuration dictionary for all nodes
NODE_CONFIG = {
    # text nodes
    "VideoCaptioning": {
        "class": VideoCaptioningNode,
        "name": "Video Captioning"
    },
    "CaptionRefinement": {
        "class": CaptionRefinementNode,
        "name": "Caption Refinement"
    }
}

def generate_node_mappings(node_config):
    """Generate node class and display name mappings from config."""
    node_class_mappings = {}
    node_display_name_mappings = {}

    for node_name, node_info in node_config.items():
        node_class_mappings[node_name] = node_info["class"]
        node_display_name_mappings[node_name] = node_info.get("name", node_info["class"].__name__)

    return node_class_mappings, node_display_name_mappings

# Generate the mappings
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)

# Export all necessary variables
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]