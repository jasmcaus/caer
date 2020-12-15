#include <c10/util/Exception.h>
#include <ATen/DynamicLibrary.h>
#include <ATen/Utils.h>

#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#else
#include <c10/util/win32-headers.h>
#endif

namespace at {


#ifndef C10_MOBILE
#ifndef _WIN32

// Unix

static void* checkDL(void* x) {
  if (!x) {
    AT_ERROR("Error in dlopen or dlsym: ", dlerror());
  }

  return x;
}
DynamicLibrary::DynamicLibrary(const char* name) {
  // NOLINTNEXTLINE(hicpp-signed-bitwise)
  handle = checkDL(dlopen(name, RTLD_LOCAL | RTLD_NOW));
}

void* DynamicLibrary::sym(const char* name) {
  AT_ASSERT(handle);
  return checkDL(dlsym(handle, name));
}

DynamicLibrary::~DynamicLibrary() {
  if (!handle)
    return;
  dlclose(handle);
}

#else

// Windows

DynamicLibrary::DynamicLibrary(const char* name) {
  // NOLINTNEXTLINE(hicpp-signed-bitwise)
  HMODULE theModule;
  bool reload = true;
  // Check if LOAD_LIBRARY_SEARCH_DEFAULT_DIRS is supported
  if (GetProcAddress(GetModuleHandle("KERNEL32.DLL"), "AddDllDirectory") != NULL) {
    theModule = LoadLibraryExA(
        name,
        NULL,
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (theModule != NULL || (GetLastError() != ERROR_MOD_NOT_FOUND)) {
      reload = false;
    }
  }

  if (reload) {
    theModule = LoadLibraryA(name);
  }

  if (theModule) {
    handle = theModule;
  } else {
    char buf[256];
    DWORD dw = GetLastError();
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                  NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                  buf, (sizeof(buf) / sizeof(char)), NULL);
    AT_ERROR("error in LoadLibrary for ", name, ". WinError ", dw, ": ", buf);
  }
}

void* DynamicLibrary::sym(const char* name) {
  AT_ASSERT(handle);
  FARPROC procAddress = GetProcAddress((HMODULE)handle, name);
  if (!procAddress) {
    AT_ERROR("error in GetProcAddress");
  }
  return (void*)procAddress;
}

DynamicLibrary::~DynamicLibrary() {
  if (!handle) {
    return;
  }
  FreeLibrary((HMODULE)handle);
}

#endif
#endif

} // namespace at
