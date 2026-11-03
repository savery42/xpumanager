/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef OAL_LOCAL_TIME_H
#define OAL_LOCAL_TIME_H

#include <chrono>
#include <string>

/**
 * @brief Formats @p when as local wall-clock time followed by its time zone.
 *
 * The zone is the abbreviation the C library has for that moment (so "PST" in winter and "PDT"
 * in summer). Zones without one, or a library that reports none, get the numeric UTC offset.
 *
 * @param[in] when  The instant to format, truncated to whole seconds.
 *
 * @retval "HH:MM:SS ZONE"   e.g. "15:42:09 PST" or "15:42:09 +03".
 * @retval "HH:MM:SS +hhmm"  The zone has no abbreviation.
 * @retval ""                @p when cannot be converted to local time.
 */
[[nodiscard]] std::string formatLocalTime(std::chrono::system_clock::time_point when);

#endif // OAL_LOCAL_TIME_H
