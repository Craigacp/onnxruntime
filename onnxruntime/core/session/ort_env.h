// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <atomic>
#include <string>
#include <mutex>
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/allocator.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
class Environment;
}

struct OrtEnv {
 public:
  struct LoggingManagerConstructionInfo {
    LoggingManagerConstructionInfo(OrtLoggingFunction logging_function1,
                                   void* logger_param1,
                                   OrtLoggingLevel default_warning_level1,
                                   const char* logid1)
        : logging_function(logging_function1),
          logger_param(logger_param1),
          default_warning_level(default_warning_level1),
          logid(logid1) {}
    OrtLoggingFunction logging_function{};
    void* logger_param{};
    OrtLoggingLevel default_warning_level;
    const char* logid{};
  };

  static OrtEnv* GetInstance(const LoggingManagerConstructionInfo& lm_info,
                             onnxruntime::common::Status& status,
                             const OrtThreadingOptions* tp_options = nullptr);

  static void Release(OrtEnv* env_ptr);

  const onnxruntime::Environment& GetEnvironment() const {
    return *value_;
  }

  onnxruntime::Environment& GetEnvironment() {
    return *value_;
  }

  onnxruntime::logging::LoggingManager* GetLoggingManager() const;
  void SetLoggingManager(std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager);

  /**
   * Registers an allocator for sharing between multiple sessions.
   * Returns an error if an allocator with the same OrtMemoryInfo is already registered.
   */
  onnxruntime::common::Status RegisterAllocator(onnxruntime::AllocatorPtr allocator);

  /**
   * Creates and registers an allocator for sharing between multiple sessions.
   * Return an error if an allocator with the same OrtMemoryInfo is already registered.
   */
  onnxruntime::common::Status CreateAndRegisterAllocator(const OrtMemoryInfo& mem_info,
                                                         const OrtArenaCfg* arena_cfg = nullptr);

  /**
   * Removes registered allocator that was previously registered for sharing between multiple sessions.
   */
  onnxruntime::common::Status UnregisterAllocator(const OrtMemoryInfo& mem_info);
  OrtEnv(std::unique_ptr<onnxruntime::Environment> value);
  ~OrtEnv();
  onnxruntime::common::Status CreateAndRegisterAllocatorV2(const std::string& provider_type, const OrtMemoryInfo& mem_info, const std::unordered_map<std::string, std::string>& options, const OrtArenaCfg* arena_cfg = nullptr);

 private:
  // p_instance_ holds the single, global instance of OrtEnv.
  // This is a raw pointer to allow for intentional memory leaking when
  // the process is shutting down (g_is_shutting_down is true).
  // Using a smart pointer like std::unique_ptr would complicate this specific
  // shutdown scenario, as it would attempt to deallocate the memory even if
  // Release() hasn't been called or if a leak is desired.
  // Management is handled by GetInstance() and Release(), with ref_count_
  // tracking active users. It is set to nullptr when the last reference is released
  // (and not shutting down).
  static OrtEnv* p_instance_;
  static std::mutex m_;
  static int ref_count_;

  std::unique_ptr<onnxruntime::Environment> value_;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OrtEnv);
};
