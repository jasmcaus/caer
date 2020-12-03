import caer 
import os 

here = os.path.dirname(__file__)

def test_target_sizes():
    test_img = os.path.join(here, 'data', 'beverages.jpg')

    img_400_400 = caer.imread(test_img, target_size=(400,400))
    img_304_339 = caer.imread(test_img, target_size=(304,339))
    img_199_206 = caer.imread(test_img, target_size=(199,206))

    assert img_400_400.shape[:2] == (400,400)
    assert img_304_339.shape[:2] == (339,304)
    assert img_199_206.shape[:2] == (206,199)


def test_keep_aspect_ratio():
    test_img = os.path.join(here, 'data', 'green_fish.jpg')

    img_400_400 = caer.imread(test_img, target_size=(400,400), keep_aspect_ratio=True)
    img_223_182 = caer.imread(test_img, target_size=(223,182), keep_aspect_ratio=True)
    img_93_35 = caer.imread(test_img, target_size=(93,35), keep_aspect_ratio=True)

    assert img_400_400.shape[:2] == (400,400)
    assert img_223_182.shape[:2] == (182,223)
    assert img_93_35.shape[:2] == (35,93)