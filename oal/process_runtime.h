/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef OAL_PROCESS_RUNTIME_H
#define OAL_PROCESS_RUNTIME_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace runtime {

/// A GPU programming interface.
enum class Api : uint8_t
{
	Sycl,
	LevelZero,
	OpenCl,
	Vulkan,
	OpenGl,
	Vpl,
	VaApi,
	Count
};

/// A library built on those interfaces.
enum class Framework : uint8_t
{
	PyTorch,
	OpenVino,
	OnnxRuntime,
	Ggml,
	FFmpeg,
	GStreamer,
	Count
};

[[nodiscard]] constexpr std::string_view label(Api api) noexcept
{
	constexpr std::array<std::string_view, static_cast<std::size_t>(Api::Count)> labels{"SYCL", "L0",  "OCL", "VK",
																						"GL",	"VPL", "VA"};
	return labels[static_cast<std::size_t>(api)];
}

[[nodiscard]] constexpr std::string_view label(Framework fw) noexcept
{
	constexpr std::array<std::string_view, static_cast<std::size_t>(Framework::Count)> labels{"Torch", "OV",	 "ORT",
																							  "GGML",  "FFmpeg", "GSt"};
	return labels[static_cast<std::size_t>(fw)];
}

/// What a process has loaded to use the GPU, and which model weight files it maps.
struct ProcessRuntime
{
	uint32_t apis = 0;					///< bit (1 << Api)
	uint32_t frameworks = 0;			///< bit (1 << Framework)
	std::vector<std::string> libraries; ///< the loaded files that set a bit, in load order
	std::vector<std::string> models;	///< mapped model weight files

	[[nodiscard]] bool has(Api api) const noexcept { return (apis & (1U << static_cast<unsigned>(api))) != 0U; }
	[[nodiscard]] bool has(Framework fw) const noexcept
	{
		return (frameworks & (1U << static_cast<unsigned>(fw))) != 0U;
	}
	bool operator==(const ProcessRuntime &) const = default;
};

} // namespace runtime

/**
 * @brief Reads which GPU APIs and frameworks a process has loaded, and the model files it maps.
 *
 * A loaded library shows what a process can drive the GPU with, not what it is driving it
 * with now. On Linux this reads /proc/<pid>/maps, which needs the process's own user or
 * CAP_SYS_PTRACE.
 *
 * @param[in] pid       Process to read.
 * @param[in] procRoot  Defaults to "/proc"; override in tests to point at a fixture directory.
 *
 * @retval runtime  What the process has loaded (empty when nothing is recognised).
 * @retval nullopt  The process has exited, may not be inspected, or the platform has no support.
 */
[[nodiscard]] std::optional<runtime::ProcessRuntime> readProcessRuntime(std::uint32_t pid,
																		const std::string &procRoot = "/proc");

#endif // OAL_PROCESS_RUNTIME_H
