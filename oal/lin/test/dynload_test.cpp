/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "dynload.h"

TEST_CASE("the driver version is the dotted suffix of the library file name")
{
	CHECK(driverVersionFromFileName("libze_intel_gpu.so.1.17.39980") == "1.17.39980");
	CHECK(driverVersionFromFileName("libze_intel_gpu.so.1.6.32224") == "1.6.32224");
}

TEST_CASE("names without a version yield nothing")
{
	CHECK_FALSE(driverVersionFromFileName("libze_intel_gpu.so.1").has_value()); // the soname
	CHECK_FALSE(driverVersionFromFileName("libze_intel_gpu.so").has_value());	// a local build
	CHECK_FALSE(driverVersionFromFileName("libze_intel_gpu.so.1.17.").has_value());
	CHECK_FALSE(driverVersionFromFileName("libze_intel_gpu.so.1.17.x").has_value());
	CHECK_FALSE(driverVersionFromFileName("libze_loader.so.1.21.9").has_value());
}
