
import gradio as gr

def CustomDropdown(**kwargs):
    """
    A wrapper around gr.Dropdown with 'Avant-Garde' styling pre-applied.
    
    Args:
        **kwargs: All standard arguments for gr.Dropdown.
    
    Returns:
        gr.Dropdown: The configured component.
    """
    # Force specific classes for the custom design
    classes = kwargs.get("elem_classes", [])
    if "custom-dropdown" not in classes:
        classes.append("custom-dropdown")
    if "premium-input" not in classes:
        classes.append("premium-input")
        
    kwargs["elem_classes"] = classes
    
    # Defaults for better UX if not specified
    if "interactive" not in kwargs:
        kwargs["interactive"] = True
        
    return gr.Dropdown(**kwargs)

def create_info_box(label: str, content: str):
    """
    Create a collapsible help box for better UX.
    
    Args:
        label (str): The label for the accordion (e.g. "Quantization").
        content (str): The Markdown content to display.
    """
    with gr.Accordion(f"❓ about {label}", open=False, elem_classes=["info-box"]):
        gr.Markdown(content, elem_classes=["info-content"])
