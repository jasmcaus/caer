rm -rf source/generated
make clean
make html --debug --jobs 2 SPHINXOPTS="-W"