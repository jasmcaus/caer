mkdir -p ~/.local/bin
export PATH=$HOME/.local/bin:$PATH
export PATH=$HOME/miniconda/bin:$PATH

if test -e $HOME/miniconda/bin; then
    echo "Miniconda is already installed."
else
    rm -rf $HOME/miniconda
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p $HOME/miniconda
    conda update --yes --quiet conda
fi

# For debugging:
conda info -a

# python -m pip install --upgrade pip wheel

# # install specific wheels from wheelhouse
# for requirement in opencv-contrib-python requests numpy; do
#     WHEELS="$WHEELS $(grep $requirement requirements/default.txt)"
# done

# # cython is not in the default.txt requirements
# WHEELS="$WHEELS $(grep -i cython requirements/build.txt)"
# python -m pip install $PIP_FLAGS $WHEELHOUSE $WHEELS

# # Install build time requirements
# python -m pip install $PIP_FLAGS -r requirements/build.txt
# # Default requirements are necessary to build because of lazy importing
# # They can be moved after the build step if #3158 is accepted
# python -m pip install $PIP_FLAGS -r requirements/default.txt

# # Show what's installed
# python -m pip list