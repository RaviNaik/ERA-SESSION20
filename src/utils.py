def loss_fn(images):
    return -images.median() / 3


concept_styles = {
    "Allante": "allante.bin",
    "XYZ": "xyz.bin",
    "Moebius": "moebius.bin",
    "Oil Style": "oil_style",
    "Polygons": "poly.bin",
}
