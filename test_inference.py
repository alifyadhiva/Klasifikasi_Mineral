from PIL import Image

from mineral_classifier.inference import preprocess_image


def test_preprocess_image_converts_to_rgb_and_resizes():
    image = Image.new("RGBA", (50, 60))
    output = preprocess_image(image, (300, 300))

    assert output.mode == "RGB"
    assert output.size == (300, 300)
