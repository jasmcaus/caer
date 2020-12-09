import os 

def test_h():
    # PATH_TO_MEDIA_FILES = os.path.dirname(os.path.join(os.getcwd(), 'media-files'))
    PATH_TO_MEDIA_FILES = os.path.join(os.path.dirname(os.getcwd()), 'media-files')
    print(PATH_TO_MEDIA_FILES)

    raise AssertionError(f'{PATH_TO_MEDIA_FILES}')

