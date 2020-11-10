from setuptools import Extension, setup
from distutils.command.build_ext import build_ext
import numpy as np 

EXTENSIONS = {
    'caer.cndi' : ['numps/cndimage.c', 
                   'numps/cndfilters.c',
                   'numps/cndfourier.c',
                   'numps/cndinterpolation.c',
                   'numps/cndmeasure.c',
                   'numps/cndmorphology.c',
                   'numps/cndsplines.c',
                   'numps/cndsupport.c'
                ]
}

EXT_MODULES = [Extension(key, sources=sources, include_dirs=[np.get_include()]) for key, sources in EXTENSIONS.items()]

copt={
    'msvc': ['/EHsc'], 
    'intelw': ['/EHsc']  
}

class build_extension_class(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c in copt:
            for e in self.extensions:
                e.extra_compile_args = copt[c]
        build_ext.build_extensions(self)

CMDCLASS = {
    'build_ext': build_extension_class
}

def setup_package():
    metadata = dict(
        ext_modules = EXT_MODULES,
        cmdclass = CMDCLASS
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()