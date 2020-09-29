Contributing Guidelines
==========================================

If you have improvements to `caer`, send us your pull requests! We'd love to accept your patches! For those just getting started, Github has a [how to](https://help.github.com/articles/using-pull-requests/)All bug fixes, new functionality, new tutorials etc. are (and should be) submitted via GitHub's mechanism of pull requests.

If you want to contribute, start working through the Caer codebase, navigate to the
[Issues](https://github.com/jasmcaus/caer/issues) tab and start looking through interesting issues. 

***NOTE***: Only original source code from you and other people that have been working on the PR can be accepted into the main repository.


Before you start contributing you should
----------------------------------------

-   Make sure you agree to the `caer` MIT license before you contribute your code

-   If you are submitting a new algorithm implementation, do a quick search over the internet to see whether the algorithm is patented or not. 

-   If you are going to fix a bug, check that it hasn't yet been spotted or there is someone working on it. In the latter, you might provide support or suggestion in the issue or in the linked pull request.

-   If you have a question about the software, then this is **NOT** the right place. Feel free to read the official [documentation](http://github.com/jasmcaus/caer/documentation.md) or if that does not help, you can contact one of the maintainers (use this as your last resort. Most of the time, we may not have the time to reply to your messages.)

Before you open up anything on the Caer GitHub page, be sure that you are at the right place with your problem.


Making a good pull request
--------------------------

Following these guidelines will increase the likelihood of your pull request being accepted:

1.  Before pushing your PR to the repository, make sure that it builds perfectly fine on your local system.
2.  Add enough information: like a meaningful title, the reason why you made the commit and a link to the issue page if you opened one for this PR.
3.  Scope your PR to one issue. Before submitting, make sure the diff contains no unrelated changes. If you want to cover more than one issue, submit your changes for each as separate pull requests.
4.  If you have added new functionality, you should update/create the relevant documentation.
5.  Try not to include "oops" commits - ones that just fix an error in the previous commit. If you have those, then before submitting [squash](http://git-scm.com/book/en/Git-Tools-Rewriting-History#Squashing-Commits) those fixes directly into the commits where they belong.
6.  Make sure to choose the right base branch and to follow the [[Coding_Style_Guide]] for your code.
7. Make sure to add test for new functionality or test that reproduces fixed bug with related test data. Please do not add extra images or videos, if some of existing media files are suitable.


The Process
---------------------------------

One of our team members will be assigned to review your pull requests. Once the pull requests are approved and pass continuous integration checks, a `ready to pull` label will be applied to your change. This means we are working on getting your pull request submitted to our internal repository. Once the change has been submitted internally, your pull request will be merged automatically on GitHub.

Last, but certainly not the least! **Make sure you get due credit**. We try to highlight all the contributions and list major ones in the [[ChangeLog]] and release announcements, but sometimes we may inadvertently forget to do that. Please do not hesitate to remind us, and we will update the ChangeLog accordingly.

Happy Contributing!