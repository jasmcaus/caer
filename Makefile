debug: caer/*.cpp caer/*.h caer/*.hpp
	DEBUG=2 python3 setup.py build --build-lib=.

fast: caer/*.cpp caer/*.h caer/*.hpp
	python setup.py build --build-lib=.

install:
	python setup.py install

fast: caer/*.cpp caer/*.h caer/*.hpp
	python3 setup.py build --build-lib=.

clean:
	rm -rf build caer/*.so caer/features/*.so

tests: debug
	pytest -v

# docs:
# 	rm -rf build/docs
# 	cd docs && make html && cp -r build/html ../build/docs
# 	@echo python setup.py upload_docs

.PHONY: clean docs tests fast debug install fast3
