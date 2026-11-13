/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 * Tests for readProcessRuntime() against fixture maps files, and once against this process.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "process_runtime.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Writes <root>/<pid>/maps and removes the tree when it goes out of scope.
class FakeProc
{
public:
	FakeProc() : root(fs::temp_directory_path() / ("xpum_process_runtime_" + std::to_string(::getpid()))) {}
	~FakeProc() { fs::remove_all(root); }
	FakeProc(const FakeProc &) = delete;
	FakeProc &operator=(const FakeProc &) = delete;
	FakeProc(FakeProc &&) = delete;
	FakeProc &operator=(FakeProc &&) = delete;

	void write(uint32_t pid, const std::string &maps) const
	{
		fs::create_directories(root / std::to_string(pid));
		std::ofstream{root / std::to_string(pid) / "maps"} << maps;
	}
	[[nodiscard]] std::string path() const { return root.string(); }

private:
	fs::path root;
};

// A maps line mapping @p path.
std::string mapLine(const std::string &path)
{
	return "7f0000000000-7f0000100000 r-xp 00000000 103:02 2222 " + path + "\n";
}

constexpr unsigned bit(runtime::Api api) { return 1U << static_cast<unsigned>(api); }

} // namespace

TEST_CASE("readProcessRuntime sorts libraries into APIs and frameworks, whatever their version")
{
	const FakeProc proc;
	proc.write(42, mapLine("/usr/bin/llama-server") +
					   "55d0c0c00000-55d0c0d00000 rw-p 00000000 00:00 0                    [heap]\n" +
					   mapLine("/usr/lib/libggml-base.so.0.25.3") + mapLine("/usr/lib/libggml-vulkan.so.0.25.3") +
					   mapLine("/usr/lib/libvulkan.so.1.4.363") + mapLine("/usr/lib/libvulkan.so.1.4.363") +
					   "7f0000200000-7f0000300000 rw-p 00000000 00:00 0\n" + mapLine("/usr/lib/libvulkan_intel.so") +
					   mapLine("/usr/lib/libc.so.6"));
	const auto rt = readProcessRuntime(42, proc.path());
	REQUIRE(rt.has_value());
	CHECK(rt->apis == bit(runtime::Api::Vulkan));
	CHECK(rt->has(runtime::Framework::Ggml));
	// Each file once, in load order; the executable, libc and anonymous mappings are not listed.
	CHECK(rt->libraries == std::vector<std::string>{"/usr/lib/libggml-base.so.0.25.3",
													"/usr/lib/libggml-vulkan.so.0.25.3",
													"/usr/lib/libvulkan.so.1.4.363", "/usr/lib/libvulkan_intel.so"});
	CHECK(rt->models.empty());
}

TEST_CASE("readProcessRuntime keeps model files apart and does not mistake look-alike libraries")
{
	const FakeProc proc;
	proc.write(7, mapLine("/models/Qwen3-8B-Q4_K_M.GGUF") + mapLine("/models/My Model.safetensors (deleted)") +
					  mapLine("/usr/lib/libglib-2.0.so.0") + mapLine("/usr/lib/libvala.so") +
					  mapLine("/usr/lib/libGLX_mesa.so.0") + mapLine("/usr/lib/libze_loader.so.1.34.0") +
					  mapLine("/usr/lib/libsycl.so.8"));
	const auto rt = readProcessRuntime(7, proc.path());
	REQUIRE(rt.has_value());
	CHECK(rt->models == std::vector<std::string>{"/models/Qwen3-8B-Q4_K_M.GGUF", "/models/My Model.safetensors"});
	CHECK(rt->apis == (bit(runtime::Api::OpenGl) | bit(runtime::Api::LevelZero) | bit(runtime::Api::Sycl)));
	CHECK(rt->frameworks == 0U);
}

TEST_CASE("readProcessRuntime tells an unreadable process from one that loads nothing")
{
	const FakeProc proc;
	CHECK_FALSE(readProcessRuntime(99, proc.path()).has_value());
	proc.write(2, "");
	const auto none = readProcessRuntime(2, proc.path());
	REQUIRE(none.has_value());
	CHECK(*none == runtime::ProcessRuntime{});
}

TEST_CASE("readProcessRuntime reads this process")
{
	CHECK(readProcessRuntime(static_cast<uint32_t>(::getpid())).has_value());
}
