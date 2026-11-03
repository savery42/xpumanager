/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "process_control.h"

#include <cerrno>
#include <csignal>
#include <limits>
#include <sys/types.h>

SignalResult sendStopSignal(std::uint32_t pid, StopSignal sig) noexcept
{
	if (pid == 0 || pid > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
		return SignalResult::Failed;
	}
	if (kill(static_cast<pid_t>(pid), sig == StopSignal::Kill ? SIGKILL : SIGTERM) == 0) {
		return SignalResult::Sent;
	}
	switch (errno) {
	case EPERM:
		return SignalResult::NotPermitted;
	case ESRCH:
		return SignalResult::NoSuchProcess;
	default:
		return SignalResult::Failed;
	}
}
