This file gives an overview of what is necessary to build binary releases for
caer.

Current build and release info
==============================
The current info on building and releasing caer and SciPy is scattered in
several places. It should be summarized in one place, updated, and where
necessary described in more detail. The sections below list all places where
useful info can be found.


Source tree
-----------
- INSTALL.rst.txt
- release.sh
- pavement.py


caer Docs
----------
- https://github.com/caer/caer/blob/master/doc/HOWTO_RELEASE.rst.txt


SciPy.org wiki
--------------
- https://www.scipy.org/Installing_SciPy and links on that page.


Release Scripts
---------------
- https://github.com/caer/caer-vendor


Supported platforms and versions
================================
:ref:`NEP 29 <NEP29>` outlines which Python versions
are supported; For the first half of 2020, this will be Python >= 3.6. We test
caer against all these versions every time we merge code to master.  Binary
installers may be available for a subset of these versions (see below).

OS X
----
OS X versions >= 10.9 are supported, for Python version support see
:ref:`NEP 29 <NEP29>`. We build binary wheels for
OSX that are compatible with Python.org Python, system Python, homebrew and
macports - see this `OSX wheel building summary
<https://github.com/MacPython/wiki/wiki/Spinning-wheels>`_ for details.


Windows
-------
We build 32- and 64-bit wheels on Windows. Windows 7, 8 and 10 are supported.
We build caer using the `mingw-w64 toolchain`_ on Appveyor.


Linux
-----
We build and ship `manylinux1 <https://www.python.org/dev/peps/pep-0513>`_
wheels for caer.  Many Linux distributions include their own binary builds
of caer.


BSD / Solaris
-------------
No binaries are provided, but successful builds on Solaris and BSD have been
reported.


Tool chain
==========
We build all our wheels on cloud infrastructure - so this list of compilers is
for information and debugging builds locally.  See the ``.travis.yml`` and
``appveyor.yml`` scripts in the `caer wheels`_ repo for the definitive source
of the build recipes. Packages that are available using pip are noted.


Compilers
---------
The same gcc version is used as the one with which Python itself is built on
each platform. At the moment this means:

- OS X builds on travis currently use `clang`.  It appears that binary wheels
  for OSX >= 10.6 can be safely built from the travis-ci OSX 10.9 VMs
  when building against the Python from the Python.org installers;
- Windows builds use the `mingw-w64 toolchain`_;
- Manylinux1 wheels use the gcc provided on the Manylinux docker images.

You will need Cython for building the binaries.  Cython compiles the ``.pyx``
files in the caer distribution to ``.c`` files.

.. _mingw-w64 toolchain : https://mingwpy.github.io

OpenBLAS
------------
All the wheels link to a version of OpenBLAS_ supplied via the openblas-libs_ repo.
The shared object (or DLL) is shipped with in the wheel, renamed to prevent name
collisions with other OpenBLAS shared objects that may exist in the filesystem.

.. _OpenBLAS: https://github.com/xianyi/OpenBLAS
.. _openblas-libs: https://github.com/MacPython/openblas-libs


Building source archives and wheels
-----------------------------------
You will need write permission for caer-wheels in order to trigger wheel
builds.

- Python(s) from `python.org <https://python.org>`_ or linux distro.
- cython (pip)
- virtualenv (pip)
- Paver (pip)
- pandoc `pandoc.org <https://www.pandoc.org>`_ or linux distro.
- caer-wheels `<https://github.com/MacPython/caer-wheels>`_ (clone)


Building docs
-------------
Building the documents requires a number of latex ``.sty`` files. Install them
all to avoid aggravation.

- Sphinx (pip)
- caerdoc (pip)
- Matplotlib
- Texlive (or MikTeX on Windows)


Uploading to PyPI
-----------------
- terryfy `<https://github.com/MacPython/terryfy>`_ (clone).
- beautifulsoup4 (pip)
- delocate (pip)
- auditwheel (pip)
- twine (pip)


Generating author/pr lists
--------------------------
You will need a personal access token
`<https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/>`_
so that scripts can access the github caer repository.

- gitpython (pip)
- pygithub (pip)


Virtualenv
----------
Virtualenv is a very useful tool to keep several versions of packages around.
It is also used in the Paver script to build the docs.


What is released
================

Wheels
------
We currently support Python 3.6-3.8 on Windows, OSX, and Linux

* Windows: 32-bit and 64-bit wheels built using Appveyor;
* OSX: x64_86 OSX wheels built using travis-ci;
* Linux: 32-bit and 64-bit Manylinux1 wheels built using travis-ci.

See the `caer wheels`_ building repository for more detail.

.. _caer wheels : https://github.com/MacPython/caer-wheels


Other
-----
- Release Notes
- Changelog


Source distribution
-------------------
We build source releases in both .zip and .tar.gz formats.


Release process
===============

Agree on a release schedule
---------------------------
A typical release schedule is one beta, two release candidates and a final
release.  It's best to discuss the timing on the mailing list first, in order
for people to get their commits in on time, get doc wiki edits merged, etc.
After a date is set, create a new maintenance/x.y.z branch, add new empty
release notes for the next version in the master branch and update the Trac
Milestones.


Make sure current branch builds a package correctly
---------------------------------------------------
::

    git clean -fxd
    python setup.py bdist
    python setup.py sdist

To actually build the binaries after everything is set up correctly, the
release.sh script can be used. For details of the build process itself, it is
best to read the pavement.py script.

.. note:: The following steps are repeated for the beta(s), release
   candidates(s) and the final release.


Check deprecations
------------------
Before the release branch is made, it should be checked that all deprecated
code that should be removed is actually removed, and all new deprecations say
in the docstring or deprecation warning at what version the code will be
removed.

Check the C API version number
------------------------------
The C API version needs to be tracked in three places

- caer/core/setup_common.py
- caer/core/code_generators/cversions.txt
- caer/core/include/caer/caerconfig.h

There are three steps to the process.

1. If the API has changed, increment the C_API_VERSION in setup_common.py. The
   API is unchanged only if any code compiled against the current API will be
   backward compatible with the last released caer version. Any changes to
   C structures or additions to the public interface will make the new API
   not backward compatible.

2. If the C_API_VERSION in the first step has changed, or if the hash of
   the API has changed, the cversions.txt file needs to be updated. To check
   the hash, run the script caer/core/cversions.py and note the API hash that
   is printed. If that hash does not match the last hash in
   caer/core/code_generators/cversions.txt the hash has changed. Using both
   the appropriate C_API_VERSION and hash, add a new entry to cversions.txt.
   If the API version was not changed, but the hash differs, you will need to
   comment out the previous entry for that API version. For instance, in caer
   1.9 annotations were added, which changed the hash, but the API was the
   same as in 1.8. The hash serves as a check for API changes, but it is not
   definitive.

   If steps 1 and 2 are done correctly, compiling the release should not give
   a warning "API mismatch detect at the beginning of the build".

3. The caer/core/include/caer/caerconfig.h will need a new
   NPY_X_Y_API_VERSION macro, where X and Y are the major and minor version
   numbers of the release. The value given to that macro only needs to be
   increased from the previous version if some of the functions or macros in
   the include files were deprecated.

The C ABI version number in caer/core/setup_common.py should only be
updated for a major release.


Check the release notes
-----------------------
Use `towncrier`_ to build the release note and
commit the changes. This will remove all the fragments from
``doc/release/upcoming_changes`` and add ``doc/release/<version>-note.rst``.
Note that currently towncrier must be installed from its master branch as the
last release (19.2.0) is outdated.

    towncrier --version "<version>"
    git commit -m"Create release note"

Check that the release notes are up-to-date.

Update the release notes with a Highlights section. Mention some of the
following:

  - major new features
  - deprecated and removed features
  - supported Python versions
  - for SciPy, supported caer version(s)
  - outlook for the near future

.. _towncrier: https://github.com/hawkowl/towncrier


Update the release status and create a release "tag"
----------------------------------------------------
Identify the commit hash of the release, e.g. 1b2e1d63ff.

::
    git co 1b2e1d63ff # gives warning about detached head

First, change/check the following variables in ``pavement.py`` depending on the
release version::

    RELEASE_NOTES = 'doc/release/1.7.0-notes.rst'
    LOG_START = 'v1.6.0'
    LOG_END = 'maintenance/1.7.x'

Do any other changes. When you are ready to release, do the following
changes::

    diff --git a/setup.py b/setup.py
    index b1f53e3..8b36dbe 100755
    --- a/setup.py
    +++ b/setup.py
    @@ -57,7 +57,7 @@ PLATFORMS           = ["Windows", "Linux", "Solaris", "Mac OS-
     MAJOR               = 1
     MINOR               = 7
     MICRO               = 0
    -ISRELEASED          = False
    +ISRELEASED          = True
     VERSION             = '%d.%d.%drc1' % (MAJOR, MINOR, MICRO)

     # Return the git revision as a string

And make sure the ``VERSION`` variable is set properly.

Now you can make the release commit and tag.  We recommend you don't push
the commit or tag immediately, just in case you need to do more cleanup. We
prefer to defer the push of the tag until we're confident this is the exact
form of the released code (see: :ref:`push-tag-and-commit`):

    git commit -s -m "REL: Release." setup.py
    git tag -s <version>

The ``-s`` flag makes a PGP (usually GPG) signed tag.  Please do sign the
release tags.

The release tag should have the release number in the annotation (tag
message).  Unfortunately, the name of a tag can be changed without breaking the
signature, the contents of the message cannot.

See: https://github.com/scipy/scipy/issues/4919 for a discussion of signing
release tags, and https://keyring.debian.org/creating-key.html for instructions
on creating a GPG key if you do not have one.

To make your key more readily identifiable as you, consider sending your key
to public keyservers, with a command such as::

    gpg --send-keys <yourkeyid>


Update the version of the master branch
---------------------------------------
Increment the release number in setup.py. Release candidates should have "rc1"
(or "rc2", "rcN") appended to the X.Y.Z format.

Also create a new version hash in cversions.txt and a corresponding version
define NPY_x_y_API_VERSION in caerconfig.h


Trigger the wheel builds
------------------------
See the `MacPython/caer wheels` repository.

In that repository edit the files:

- ``azure/posix.yml``
- ``azure/windows.yml``.

In both cases, set the ``BUILD_COMMIT`` variable to the current release tag -
e.g. ``v1.19.0``::

    $ gvim azure/posix.yml azure/windows.yml
    $ git commit -a
    $ git push upstream HEAD

Make sure that the release tag has been pushed.

Trigger a build by pushing a commit of your edits to the repository. Note that
you can do this on a branch, but it must be pushed upstream to the
``MacPython/caer-wheels`` repository to trigger uploads since only
that repo has the appropriate tokens to allow uploads.

The wheels, once built, appear at https://anaconda.org/multibuild-wheels-staging/caer

Make the release
----------------
Build the changelog and notes for upload with::

    paver write_release


Build and archive documentation
-------------------------------
Do::

    cd doc/
    make dist

to check that the documentation is in a buildable state. Then, after tagging,
create an archive of the documentation in the caer/doc repo::

    # This checks out github.com/caer/doc and adds (``git add``) the
    # documentation to the checked out repo.
    make merge-doc
    # Now edit the ``index.html`` file in the repo to reflect the new content.
    # If the documentation is for a non-patch release (e.g. 1.19 -> 1.20),
    # make sure to update the ``stable`` symlink to point to the new directory.
    ln -sfn <latest_stable_directory> stable
    # Commit the changes
    git -C build/merge commit -am "Add documentation for <version>"
    # Push to caer/doc repo
    git -C build/merge push


Update PyPI
-----------
The wheels and source should be uploaded to PyPI.

You should upload the wheels first, and the source formats last, to make sure
that pip users don't accidentally get a source install when they were
expecting a binary wheel.

You can do this automatically using the ``wheel-uploader`` script from
https://github.com/MacPython/terryfy.  Here is the recommended incantation for
downloading all the Windows, Manylinux, OSX wheels and uploading to PyPI. ::

    NPY_WHLS=~/wheelhouse   # local directory to cache wheel downloads
    CDN_URL=https://anaconda.org/multibuild-wheels-staging/caer/files
    wheel-uploader -u $CDN_URL -w $NPY_WHLS -v -s -t win caer 1.11.1rc1
    wheel-uploader -u $CDN_URL -w warehouse -v -s -t macosx caer 1.11.1rc1
    wheel-uploader -u $CDN_URL -w warehouse -v -s -t manylinux1 caer 1.11.1rc1

The ``-v`` flag gives verbose feedback, ``-s`` causes the script to sign the
wheels with your GPG key before upload. Don't forget to upload the wheels
before the source tarball, so there is no period for which people switch from
an expected binary install to a source install from PyPI.

There are two ways to update the source release on PyPI, the first one is::

    $ git clean -fxd  # to be safe
    $ python setup.py sdist --formats=gztar,zip  # to check
    # python setup.py sdist --formats=gztar,zip upload --sign

This will ask for your key PGP passphrase, in order to sign the built source
packages.

The second way is to upload the PKG_INFO file inside the sdist dir in the
web interface of PyPI. The source tarball can also be uploaded through this
interface.

.. _push-tag-and-commit:


Push the release tag and commit
-------------------------------
Finally, now you are confident this tag correctly defines the source code that
you released you can push the tag and release commit up to github::

    git push  # Push release commit
    git push upstream <version>  # Push tag named <version>

where ``upstream`` points to the main https://github.com/caer/caer.git
repository.


Update scipy.org
----------------
A release announcement with a link to the download site should be placed in the
sidebar of the front page of scipy.org.

The scipy.org should be a PR at https://github.com/scipy/scipy.org. The file
that needs modification is ``www/index.rst``. Search for ``News``.


Announce to the lists
---------------------
The release should be announced on the mailing lists of
caer and SciPy, to python-announce, and possibly also those of
Matplotlib, IPython and/or Pygame.

During the beta/RC phase, an explicit request for testing the binaries with
several other libraries (SciPy/Matplotlib/Pygame) should be posted on the
mailing list.


Announce to Linux Weekly News
-----------------------------
Email the editor of LWN to let them know of the release.  Directions at:
https://lwn.net/op/FAQ.lwn#contact


After the final release
-----------------------
After the final release is announced, a few administrative tasks are left to be
done:

  - Forward port changes in the release branch to release notes and release
    scripts, if any, to master branch.
  - Update the Milestones in Trac.
