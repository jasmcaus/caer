#include <c10/util/Exception.h>
#include <c10/util/Backtrace.h>
#include <c10/util/Type.h>
#include <c10/util/Logging.h>

#include <iostream>
#include <sstream>
#include <numeric>
#include <string>

namespace c10 {

Error::Error(std::string msg, std::string backtrace, const void* caller)
    : msg_(std::move(msg)), backtrace_(std::move(backtrace)), caller_(caller) {
  refresh_what();
}

// PyTorch-style error message
// Error::Error(SourceLocation source_location, const std::string& msg)
// NB: This is defined in Logging.cpp for access to GetFetchStackTrace

// Caffe2-style error message
Error::Error(
    const char* file,
    const uint32_t line,
    const char* condition,
    const std::string& msg,
    const std::string& backtrace,
    const void* caller)
    : Error(
          str("[enforce fail at ",
              detail::StripBasename(file),
              ":",
              line,
              "] ",
              condition,
              ". ",
              msg),
          backtrace,
          caller) {}

std::string Error::compute_what(bool include_backtrace) const {
  std::ostringstream oss;

  oss << msg_;

  if (context_.size() == 1) {
    // Fold error and context in one line
    oss << " (" << context_[0] << ")";
  } else {
    for (const auto& c : context_) {
      oss << "\n  " << c;
    }
  }

  if (include_backtrace) {
    oss << "\n" << backtrace_;
  }

  return oss.str();
}

void Error::refresh_what() {
  what_ = compute_what(/*include_backtrace*/ true);
  what_without_backtrace_ = compute_what(/*include_backtrace*/ false);
}

void Error::add_context(std::string new_msg) {
  context_.push_back(std::move(new_msg));
  // TODO: Calling add_context O(n) times has O(n^2) cost.  We can fix
  // this perf problem by populating the fields lazily... if this ever
  // actually is a problem.
  // NB: If you do fix this, make sure you do it in a thread safe way!
  // what() is almost certainly expected to be thread safe even when
  // accessed across multiple threads
  refresh_what();
}

namespace detail {

void torchCheckFail(const char *func, const char *file, uint32_t line, const std::string& msg) {
  throw ::c10::Error({func, file, line}, msg);
}

} // namespace detail

namespace Warning {

namespace {
  WarningHandler* getBaseHandler() {
    static WarningHandler base_warning_handler_ = WarningHandler();
    return &base_warning_handler_;
  };

  class ThreadWarningHandler {
    public:
      ThreadWarningHandler() = delete;

      static WarningHandler* get_handler() {
        if (!warning_handler_) {
          warning_handler_ = getBaseHandler();
        }
        return warning_handler_;
      }

      static void set_handler(WarningHandler* handler) {
        warning_handler_ = handler;
      }

    private:
      static thread_local WarningHandler* warning_handler_;
  };

  thread_local WarningHandler* ThreadWarningHandler::warning_handler_ = nullptr;

}

void warn(SourceLocation source_location, const std::string& msg, const bool verbatim) {
  ThreadWarningHandler::get_handler()->process(source_location, msg, verbatim);
}

void set_warning_handler(WarningHandler* handler) noexcept(true) {
  ThreadWarningHandler::set_handler(handler);
}

WarningHandler* get_warning_handler() noexcept(true) {
  return ThreadWarningHandler::get_handler();
}

} // namespace Warning

void WarningHandler::process(
    const SourceLocation& source_location,
    const std::string& msg,
    const bool /*verbatim*/) {
  LOG_AT_FILE_LINE(WARNING, source_location.file, source_location.line)
      << "Warning: " << msg << " (function " << source_location.function << ")";
}


std::string GetExceptionString(const std::exception& e) {
#ifdef __GXX_RTTI
  return demangle(typeid(e).name()) + ": " + e.what();
#else
  return std::string("Exception (no RTTI available): ") + e.what();
#endif // __GXX_RTTI
}

} // namespace c10
