/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * ONNX Runtime Java API.
 */
module ai.onnxruntime {
    requires java.logging;

    exports ai.onnxruntime;
    exports ai.onnxruntime.providers;
    exports ai.onnxruntime.platform;
}