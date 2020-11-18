import caer 
import os 

here = os.path.dirname(__file__)

def test_gray():
    test_img = os.path.join(here, 'data', 'blue_tang.png')

    img = caer.imread(test_img, channels=1)

    assert len(img.shape) == 2


