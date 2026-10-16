/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "dynload.h"
#include <algorithm>
#include <cctype>
#include <dlfcn.h>
#include <filesystem>
#include <link.h>
#include <system_error>

void *resolveL0Sym(const char *name) noexcept
{
	// RTLD_NOLOAD: attach to the already-loaded Level Zero loader without
	// triggering a fresh load, and without searching LD_PRELOAD-injected
	// libraries. Returns nullptr (gracefully) if ze_loader is not mapped.
	static void *handle = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_NOLOAD);
	if (handle == nullptr) {
		return nullptr;
	}
	return dlsym(handle, name);
}

std::optional<std::string> driverVersionFromFileName(std::string_view fileName)
{
	constexpr std::string_view prefix = "libze_intel_gpu.so.";
	if (!fileName.starts_with(prefix)) {
		return std::nullopt;
	}
	const std::string_view version = fileName.substr(prefix.size());
	const bool dottedNumbers =
		!version.empty() && std::isdigit(static_cast<unsigned char>(version.front())) &&
		std::isdigit(static_cast<unsigned char>(version.back())) &&
		std::ranges::all_of(version, [](unsigned char c) { return c == '.' || std::isdigit(c); });
	// "1" alone is the soname, not a version.
	if (!dottedNumbers || version.find('.') == std::string_view::npos) {
		return std::nullopt;
	}
	return std::string{version};
}

std::optional<std::string> loadedGpuDriverVersion()
{
	void *handle = dlopen("libze_intel_gpu.so.1", RTLD_NOW | RTLD_NOLOAD);
	if (handle == nullptr) {
		return std::nullopt;
	}
	link_map *map = nullptr;
	const bool found = dlinfo(handle, RTLD_DI_LINKMAP, &map) == 0 && map != nullptr && map->l_name != nullptr;
	const std::string path = found ? map->l_name : "";
	dlclose(handle); // drops only the reference RTLD_NOLOAD added
	std::error_code ec;
	const std::filesystem::path resolved = std::filesystem::canonical(path, ec);
	return ec ? std::nullopt : driverVersionFromFileName(resolved.filename().string());
}
