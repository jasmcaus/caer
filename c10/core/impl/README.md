c10/core/impl provides headers for functionality that is only needed in very
*specific* use-cases (e.g., you are defining a new device type), which are
generally only needed by C10 or PyTorch code.  If you are an ordinary end-user,
you **should not** use headers in this folder.  We permanently give NO
backwards-compatibility guarantees for implementations in this folder.

Compare with [c10/util](c10/util), which provides functionality that is not
directly related to being a deep learning library (e.g., C++20 polyfills), but
may still be generally useful and visible to users.

(We don't call this c10/detail, because the detail namespace convention is for
*header private* details.  However, c10::impl may be utilized from external
headers; it simply indicates that the functionality is not for end users.)
