/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef OAL_PROCESS_CONTROL_H
#define OAL_PROCESS_CONTROL_H

#include <cstdint>

enum class StopSignal : std::uint8_t
{
	Terminate, ///< ask the process to exit (SIGTERM)
	Kill,	   ///< end it at once (SIGKILL)
};

enum class SignalResult : std::uint8_t
{
	Sent,
	NotPermitted,  ///< the caller may not signal this process, e.g. another user's
	NoSuchProcess, ///< it has already exited
	Failed,		   ///< any other error, an invalid pid, or no support on this platform
};

/**
 * @brief Sends @p sig to process @p pid.
 *
 * @param[in] pid  Process to signal. 0 and values above INT_MAX are rejected:
 *                 kill(0, ...) would signal the caller's whole process group.
 * @param[in] sig  Which signal.
 *
 * @retval Sent           The signal was delivered.
 * @retval NotPermitted   EPERM.
 * @retval NoSuchProcess  ESRCH.
 * @retval Failed         Anything else; always on Windows.
 */
[[nodiscard]] SignalResult sendStopSignal(std::uint32_t pid, StopSignal sig) noexcept;

#endif // OAL_PROCESS_CONTROL_H
