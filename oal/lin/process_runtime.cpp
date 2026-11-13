/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "process_runtime.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>

namespace {

using runtime::Api;
using runtime::Framework;

// A maps line is "address perms offset dev inode pathname"; the path is the sixth field and
// may itself hold spaces.
constexpr std::size_t FIELDS_BEFORE_PATH = 5;
constexpr std::string_view DELETED_MARK = " (deleted)";

/// The path a maps line names; empty for an anonymous or pseudo mapping such as [heap]. A file
/// deleted since it was mapped keeps its path without the " (deleted)" mark.
[[nodiscard]] std::string_view pathOf(std::string_view line) noexcept
{
	for (std::size_t field = 0; field < FIELDS_BEFORE_PATH; ++field) {
		const auto start = line.find_first_not_of(' ');
		if (start == std::string_view::npos) {
			return {};
		}
		line.remove_prefix(start);
		const auto end = line.find(' ');
		if (end == std::string_view::npos) {
			return {};
		}
		line.remove_prefix(end);
	}
	const auto start = line.find_first_not_of(' ');
	if (start == std::string_view::npos) {
		return {};
	}
	line.remove_prefix(start);
	if (!line.starts_with('/')) {
		return {};
	}
	if (line.ends_with(DELETED_MARK)) {
		line.remove_suffix(DELETED_MARK.size());
	}
	return line;
}

template <typename T> struct Pattern
{
	std::string_view name; ///< library name with its version suffix removed
	bool prefix;		   ///< match names starting with @c name, not only @c name itself
	T what;
};

// Library names are compared lower-cased, with any ".so.<version>" cut back to ".so".
constexpr std::array API_PATTERNS{
	Pattern<Api>{"libsycl.so", false, Api::Sycl},
	Pattern<Api>{"libze_loader.so", false, Api::LevelZero},
	Pattern<Api>{"libze_intel_gpu.so", false, Api::LevelZero},
	Pattern<Api>{"libopencl.so", false, Api::OpenCl},
	Pattern<Api>{"libigdrcl.so", false, Api::OpenCl},
	Pattern<Api>{"libvulkan.so", false, Api::Vulkan},
	Pattern<Api>{"libvulkan_intel", true, Api::Vulkan},
	Pattern<Api>{"libgl.so", false, Api::OpenGl},
	Pattern<Api>{"libglx", true, Api::OpenGl},
	Pattern<Api>{"libegl", true, Api::OpenGl},
	Pattern<Api>{"libgles", true, Api::OpenGl},
	Pattern<Api>{"libgallium", true, Api::OpenGl},
	Pattern<Api>{"iris_dri.so", false, Api::OpenGl},
	Pattern<Api>{"libvpl.so", false, Api::Vpl},
	Pattern<Api>{"libmfx-gen.so", false, Api::Vpl},
	Pattern<Api>{"libmfxhw64.so", false, Api::Vpl},
	Pattern<Api>{"libva.so", false, Api::VaApi},
	Pattern<Api>{"libva-", true, Api::VaApi},
};

constexpr std::array FRAMEWORK_PATTERNS{
	Pattern<Framework>{"libtorch", true, Framework::PyTorch},
	Pattern<Framework>{"libintel-ext-pt", true, Framework::PyTorch},
	Pattern<Framework>{"libopenvino", true, Framework::OpenVino},
	Pattern<Framework>{"libonnxruntime", true, Framework::OnnxRuntime},
	Pattern<Framework>{"libggml", true, Framework::Ggml},
	Pattern<Framework>{"libavcodec.so", false, Framework::FFmpeg},
	Pattern<Framework>{"libgstreamer-1.0.so", false, Framework::GStreamer},
};

constexpr std::array<std::string_view, 7> MODEL_EXTENSIONS{".gguf", ".safetensors", ".onnx",  ".pt",
														   ".pth",	".blob",		".tflite"};

/// The file name of @p path, lower-cased.
[[nodiscard]] std::string baseName(std::string_view path)
{
	const auto slash = path.rfind('/');
	std::string name{slash == std::string_view::npos ? path : path.substr(slash + 1)};
	std::ranges::transform(name, name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return name;
}

/// "libvulkan.so.1.4.363" -> "libvulkan.so"; names without ".so." are returned unchanged.
[[nodiscard]] std::string_view withoutVersion(std::string_view name) noexcept
{
	const auto so = name.find(".so.");
	return so == std::string_view::npos ? name : name.substr(0, so + 3);
}

template <typename T, std::size_t N>
[[nodiscard]] bool matchInto(std::string_view lib, const std::array<Pattern<T>, N> &patterns, uint32_t &bits) noexcept
{
	for (const auto &p : patterns) {
		if (p.prefix ? lib.starts_with(p.name) : lib == p.name) {
			bits |= 1U << static_cast<unsigned>(p.what);
			return true;
		}
	}
	return false;
}

/// Adds one mapped file to @p out: a model file, a recognised library, or neither.
void classify(const std::string &path, runtime::ProcessRuntime &out)
{
	const std::string name = baseName(path);
	if (std::ranges::any_of(MODEL_EXTENSIONS, [&](std::string_view ext) { return name.ends_with(ext); })) {
		out.models.push_back(path);
		return;
	}
	const std::string_view lib = withoutVersion(name);
	const bool api = matchInto(lib, API_PATTERNS, out.apis);
	const bool fw = matchInto(lib, FRAMEWORK_PATTERNS, out.frameworks);
	if (api || fw) {
		out.libraries.push_back(path);
	}
}

} // namespace

std::optional<runtime::ProcessRuntime> readProcessRuntime(std::uint32_t pid, const std::string &procRoot)
{
	// The kernel checks access when the file is opened, so another user's process fails here.
	std::ifstream in{procRoot + "/" + std::to_string(pid) + "/maps"};
	if (!in) {
		return std::nullopt;
	}
	runtime::ProcessRuntime out;
	std::unordered_set<std::string> seen;
	std::string line;
	while (std::getline(in, line)) {
		const std::string_view path = pathOf(line);
		if (!path.empty() && seen.emplace(path).second) {
			classify(std::string{path}, out);
		}
	}
	return out;
}
