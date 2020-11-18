if test -e $HOME/miniconda/envs/condaenv; then
    echo "Condaenv already exists"
else
    conda create  --quiet --yes -n condaenv python=${TRAVIS_PYTHON_VERSION}
    conda install --quiet --yes -n condaenv opencv-contrib-python numpy requests pytest pip coveralls
fi

source activate condaenv
make debug