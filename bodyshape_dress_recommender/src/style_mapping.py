import os

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
IMAGES_DIR = os.path.abspath(IMAGES_DIR)

def get_style_recommendation(shape):
    """Return (text_recommendation, [list_of_image_paths])."""
    shape = shape.title().replace(" ", "")

    # mapping from shape → (text, [filenames])
    RECS = {
        "Pear": (
            "A-line skirts, off-shoulder tops, and detailed upper-body pieces to balance hips.",
            ["pear1.jpg", "pear2.jpg","pear3.jpg"]
        ),
        "InvertedTriangle": (
            "V-neck tops, flared pants, and soft-shoulder fabrics to reduce broadness.",
            ["invtri1.jpg", "invtri2.jpg"]
        ),
        "Hourglass": (
            "Wrap dresses, pencil skirts, and fitted tops that accentuate the waist.",
            ["hourglass1.jpg", "hourglass2.jpg"]
        ),
        "Apple": (
            "Empire waist dresses, flowy tops, and vertical patterns to elongate your torso.",
            ["apple1.jpg", "apple2.jpg"]
        ),
        "Rectangle": (
            "Ruffle tops, belted dresses, and layered outfits to add curves.",
            ["rectangle1.jpg", "rectangle2.jpg"]
        ),
    }

    text, files = RECS.get(shape, (
        "Try clothes that emphasize your best features and boost comfort.",
        ["default1.jpg"]
    ))

    # build absolute paths: images/<Shape>/<filename>
    image_paths = [os.path.join(IMAGES_DIR, shape, f) for f in files]
    return text, image_paths