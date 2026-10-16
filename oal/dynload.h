/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef OAL_DYNLOAD_H
#define OAL_DYNLOAD_H

#include <optional>
#include <string>
#include <string_view>

/**
 * @brief Resolves a symbol from the Level Zero runtime library.
 *
 * Searches the already-loaded Level Zero loader for @p name and returns its
 * address as a @c void*, or @c nullptr if the symbol is not present (e.g. the
 * installed runtime is older than the SDK version that introduced it).
 * Callers must use @c std::bit_cast to convert the result to the target
 * function-pointer type; @c reinterpret_cast between data and function
 * pointers is undefined behaviour.
 *
 * @param name  Null-terminated symbol name.
 * @return Address of the symbol, or @c nullptr if not found.
 */
[[nodiscard]] void *resolveL0Sym(const char *name) noexcept;

/**
 * @brief Version of the already-loaded Level Zero GPU driver library, from its file name.
 *
 * The compute runtime installs its library as libze_intel_gpu.so.<major>.<minor>.<build>,
 * built from the same numbers zeDriverGetApiVersion() and ze_driver_properties_t report, so
 * the version is known once zesInit has loaded the library, without zeInit.
 *
 * @retval "1.17.39980"  The library is loaded and its resolved file name carries a version.
 * @retval nullopt       Not loaded, an unversioned file name (a local build), or Windows.
 */
[[nodiscard]] std::optional<std::string> loadedGpuDriverVersion();

/**
 * @brief The version suffix of a driver library file name.
 *
 * @param[in] fileName  E.g. "libze_intel_gpu.so.1.17.39980".
 * @retval "1.17.39980"  @p fileName is the driver library followed by dotted numbers.
 * @retval nullopt       Otherwise.
 */
[[nodiscard]] std::optional<std::string> driverVersionFromFileName(std::string_view fileName);

#endif // OAL_DYNLOAD_H
