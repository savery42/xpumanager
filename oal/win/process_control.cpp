/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "process_control.h"

SignalResult sendStopSignal(std::uint32_t, StopSignal) noexcept { return SignalResult::Failed; }
