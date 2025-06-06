// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// EtwSink.h must come before the windows includes
#include "core/platform/windows/logging/etw_sink.h"

#ifdef ETW_TRACE_LOGGING_SUPPORTED

// STL includes
#include <exception>

// ETW includes
// need space after Windows.h to prevent clang-format re-ordering breaking the build.
// TraceLoggingProvider.h must follow Windows.h
#include <Windows.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26440)  // Warning C26440 from TRACELOGGING_DEFINE_PROVIDER
#endif

#include <TraceLoggingProvider.h>
#include <evntrace.h>

// See: https://developercommunity.visualstudio.com/content/problem/85934/traceloggingproviderh-is-incompatible-with-utf-8.html
#ifdef _TlgPragmaUtf8Begin
#undef _TlgPragmaUtf8Begin
#define _TlgPragmaUtf8Begin
#endif

#ifdef _TlgPragmaUtf8End
#undef _TlgPragmaUtf8End
#define _TlgPragmaUtf8End
#endif

// Different versions of TraceLoggingProvider.h contain different macro variable names for the utf8 begin and end,
// and we need to cover the lower case version as well.
#ifdef _tlgPragmaUtf8Begin
#undef _tlgPragmaUtf8Begin
#define _tlgPragmaUtf8Begin
#endif

#ifdef _tlgPragmaUtf8End
#undef _tlgPragmaUtf8End
#define _tlgPragmaUtf8End
#endif

namespace onnxruntime {
namespace logging {

namespace {
TRACELOGGING_DEFINE_PROVIDER(etw_provider_handle, "ONNXRuntimeTraceLoggingProvider",
                             // {929DD115-1ECB-4CB5-B060-EBD4983C421D}
                             (0x929dd115, 0x1ecb, 0x4cb5, 0xb0, 0x60, 0xeb, 0xd4, 0x98, 0x3c, 0x42, 0x1d));
}  // namespace

#ifdef _MSC_VER
#pragma warning(pop)
#endif

EtwRegistrationManager& EtwRegistrationManager::Instance() {
  static EtwRegistrationManager instance;
  instance.LazyInitialize();
  return instance;
}

bool EtwRegistrationManager::SupportsETW() {
  return true;
}

bool EtwRegistrationManager::IsEnabled() const {
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return is_enabled_;
}

UCHAR EtwRegistrationManager::Level() const {
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return level_;
}

Severity EtwRegistrationManager::MapLevelToSeverity() {
  switch (level_) {
    case TRACE_LEVEL_NONE:
      return Severity::kFATAL;  // There is no none severity option
    case TRACE_LEVEL_VERBOSE:
      return Severity::kVERBOSE;
    case TRACE_LEVEL_INFORMATION:
      return Severity::kINFO;
    case TRACE_LEVEL_WARNING:
      return Severity::kWARNING;
    case TRACE_LEVEL_ERROR:
      return Severity::kERROR;
    case TRACE_LEVEL_CRITICAL:
      return Severity::kFATAL;
    default:
      return Severity::kVERBOSE;
  }
}

ULONGLONG EtwRegistrationManager::Keyword() const {
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return keyword_;
}

HRESULT EtwRegistrationManager::Status() const {
  return etw_status_;
}

void EtwRegistrationManager::RegisterInternalCallback(const std::string& cb_key, EtwInternalCallback callback) {
  std::lock_guard<std::mutex> lock(callbacks_mutex_);
  [[maybe_unused]] auto result = callbacks_.emplace(cb_key, std::move(callback));
  assert(result.second);
}

void EtwRegistrationManager::UnregisterInternalCallback(const std::string& cb_key) {
  std::lock_guard<std::mutex> lock(callbacks_mutex_);
  callbacks_.erase(cb_key);
}

void NTAPI EtwRegistrationManager::ORT_TL_EtwEnableCallback(
    _In_ LPCGUID SourceId,
    _In_ ULONG IsEnabled,
    _In_ UCHAR Level,
    _In_ ULONGLONG MatchAnyKeyword,
    _In_ ULONGLONG MatchAllKeyword,
    _In_opt_ PEVENT_FILTER_DESCRIPTOR FilterData,
    _In_opt_ PVOID CallbackContext) {
  auto& manager = EtwRegistrationManager::Instance();
  {
    std::lock_guard<std::mutex> lock(manager.provider_change_mutex_);
    manager.is_enabled_ = (IsEnabled != 0);
    manager.level_ = Level;
    manager.keyword_ = MatchAnyKeyword;
  }
  manager.InvokeCallbacks(SourceId, IsEnabled, Level, MatchAnyKeyword, MatchAllKeyword, FilterData, CallbackContext);
}

EtwRegistrationManager::EtwRegistrationManager()
    : initialization_status_(InitializationStatus::NotInitialized),
      is_enabled_(false),
      level_(),
      keyword_(0),
      etw_status_(S_OK) {
}

void EtwRegistrationManager::LazyInitialize() {
  if (initialization_status_ == InitializationStatus::NotInitialized) {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (initialization_status_ == InitializationStatus::NotInitialized) {  // Double-check locking pattern
      initialization_status_ = InitializationStatus::Initializing;
      etw_status_ = ::TraceLoggingRegisterEx(etw_provider_handle, ORT_TL_EtwEnableCallback, nullptr);
      if (FAILED(etw_status_)) {
        // Registration can fail when running under Low Integrity process, and should be non-fatal
        initialization_status_ = InitializationStatus::Failed;
        // Injection of ETW logger can happen very early if ETW provider was already listening.
        // Don't use LOGS_DEFAULT here or can get "Attempt to use DefaultLogger but none has been registered"
        std::cerr << "Error in ETW registration: " << std::to_string(etw_status_) << std::endl;
      }
      initialization_status_ = InitializationStatus::Initialized;
    }
  }
}

EtwRegistrationManager::~EtwRegistrationManager() {
  if (initialization_status_ == InitializationStatus::Initialized) {
    ::TraceLoggingUnregister(etw_provider_handle);
    initialization_status_ = InitializationStatus::NotInitialized;
  }
}

void EtwRegistrationManager::InvokeCallbacks(LPCGUID SourceId, ULONG IsEnabled, UCHAR Level, ULONGLONG MatchAnyKeyword,
                                             ULONGLONG MatchAllKeyword, PEVENT_FILTER_DESCRIPTOR FilterData,
                                             PVOID CallbackContext) {
  if (initialization_status_ != InitializationStatus::Initialized) {
    // Drop messages until manager is fully initialized.
    return;
  }

  std::lock_guard<std::mutex> lock(callbacks_mutex_);
  for (const auto& entry : callbacks_) {
    const auto& cb = entry.second;
    cb(SourceId, IsEnabled, Level, MatchAnyKeyword, MatchAllKeyword, FilterData, CallbackContext);
  }
}

void EtwSink::SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) {
  UNREFERENCED_PARAMETER(timestamp);

  // register on first usage
  static EtwRegistrationManager& etw_manager = EtwRegistrationManager::Instance();

  // do something (not that meaningful) with etw_manager so it doesn't get optimized out
  // as we want an instance around to do the unregister
  if (FAILED(etw_manager.Status())) {
    return;
  }

  // TODO: Validate if this filtering makes sense.
  if (message.DataType() == DataType::USER) {
    return;
  }

  // NOTE: Theoretically we could create an interface for all the ETW system interactions so we can separate
  // out those from the logic in this class so it is more testable.
  // Right now the logic is trivial, so that effort isn't worth it.

  // TraceLoggingWrite requires (painfully) a compile time constant for the TraceLoggingLevel,
  // forcing us to use an ugly macro for the call.
#define ETW_EVENT_NAME "ONNXRuntimeLogEvent"
#define TRACE_LOG_WRITE(level)                                                                                      \
  TraceLoggingWrite(etw_provider_handle, ETW_EVENT_NAME,                                                            \
                    TraceLoggingKeyword(static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Logs)), \
                    TraceLoggingLevel(level),                                                                       \
                    TraceLoggingString(logger_id.c_str(), "logger"),                                                \
                    TraceLoggingString(message.Category(), "category"),                                             \
                    TraceLoggingString(message.Location().ToString().c_str(), "location"),                          \
                    TraceLoggingString(message.Message().c_str(), "message"))

  const auto severity{message.Severity()};

  GSL_SUPPRESS(bounds)
  GSL_SUPPRESS(type) {
    switch (severity) {
      case Severity::kVERBOSE:
        TRACE_LOG_WRITE(TRACE_LEVEL_VERBOSE);
        break;
      case Severity::kINFO:
        TRACE_LOG_WRITE(TRACE_LEVEL_INFORMATION);
        break;
      case Severity::kWARNING:
        TRACE_LOG_WRITE(TRACE_LEVEL_WARNING);
        break;
      case Severity::kERROR:
        TRACE_LOG_WRITE(TRACE_LEVEL_ERROR);
        break;
      case Severity::kFATAL:
        TRACE_LOG_WRITE(TRACE_LEVEL_CRITICAL);
        break;
      default:
        ORT_THROW("Unexpected Severity of " + std::to_string(static_cast<int>(severity)));
    }
  }

#undef ETW_EVENT_NAME
#undef TRACE_LOG_WRITE
}
}  // namespace logging
}  // namespace onnxruntime
#else
// ETW is not supported on this platform but should still define a dummy EtwRegistrationManager
// so that it can be used in the EP provider bridge.
namespace onnxruntime {
namespace logging {
EtwRegistrationManager& EtwRegistrationManager::Instance() {
  static EtwRegistrationManager instance;
  return instance;
}

bool EtwRegistrationManager::SupportsETW() {
  return false;
}
}  // namespace logging
}  // namespace onnxruntime
#endif  // ETW_TRACE_LOGGING_SUPPORTED
