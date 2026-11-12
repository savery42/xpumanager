/*
 * Copyright (C) 2025-2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
// doctest defines INFO(expr) for test context; undef it so logger.h
// (pulled in via driver_util.h → device.h) can define INFO(fmt, ...).
#undef INFO

#include "driver_util.h"

#include <array>
#include <string_view>

using xpum::detail::deduplicateByIndex;
using xpum::detail::deviceConditionOf;
using xpum::detail::deviceLevelIndex;
using xpum::detail::deviceRecoveryActionOf;
using xpum::detail::knownDevState;
using xpum::detail::parseUint32;
using xpum::detail::resetRecoveryActionOf;

// ---------------------------------------------------------------------------
// parseUint32
// ---------------------------------------------------------------------------

TEST_CASE("parseUint32: valid decimal integers")
{
	CHECK(parseUint32("0") == std::optional<uint32_t>{0});
	CHECK(parseUint32("1") == std::optional<uint32_t>{1});
	CHECK(parseUint32("42") == std::optional<uint32_t>{42});
	CHECK(parseUint32("4294967295") == std::optional<uint32_t>{4294967295u}); // UINT32_MAX
}

TEST_CASE("parseUint32: empty or non-numeric input returns nullopt")
{
	CHECK(!parseUint32(""));
	CHECK(!parseUint32("abc"));
	CHECK(!parseUint32("0000:03:00.0")); // BDF address
	CHECK(!parseUint32("-1"));			 // negative
}

TEST_CASE("parseUint32: partial-numeric input returns nullopt")
{
	CHECK(!parseUint32("1abc")); // trailing non-digit
	CHECK(!parseUint32("1 "));	 // trailing space
	CHECK(!parseUint32(" 1"));	 // leading space
}

TEST_CASE("parseUint32: overflow returns nullopt")
{
	CHECK(!parseUint32("4294967296")); // UINT32_MAX + 1
	CHECK(!parseUint32("99999999999"));
}

// ---------------------------------------------------------------------------
// deduplicateByIndex
// ---------------------------------------------------------------------------

static devInfo makeDevInfo(uint32_t index)
{
	devInfo d{};
	d.index = index;
	return d;
}

TEST_CASE("deduplicateByIndex: no duplicates — list unchanged")
{
	std::vector<devInfo> list = {makeDevInfo(0), makeDevInfo(1), makeDevInfo(2)};
	deduplicateByIndex(list);
	REQUIRE(list.size() == 3);
	CHECK(list[0].index == 0);
	CHECK(list[1].index == 1);
	CHECK(list[2].index == 2);
}

TEST_CASE("deduplicateByIndex: duplicate removed, first occurrence kept")
{
	std::vector<devInfo> list = {makeDevInfo(0), makeDevInfo(0)};
	deduplicateByIndex(list);
	REQUIRE(list.size() == 1);
	CHECK(list[0].index == 0);
}

TEST_CASE("deduplicateByIndex: preserves insertion order")
{
	// "2,0,1,0" — second 0 removed, order 2-0-1 preserved
	std::vector<devInfo> list = {makeDevInfo(2), makeDevInfo(0), makeDevInfo(1), makeDevInfo(0)};
	deduplicateByIndex(list);
	REQUIRE(list.size() == 3);
	CHECK(list[0].index == 2);
	CHECK(list[1].index == 0);
	CHECK(list[2].index == 1);
}

TEST_CASE("deduplicateByIndex: empty list is a no-op")
{
	std::vector<devInfo> list;
	deduplicateByIndex(list);
	CHECK(list.empty());
}

// ---------------------------------------------------------------------------
// deviceLevelIndex
// ---------------------------------------------------------------------------

TEST_CASE("deviceLevelIndex: no handles gives nothing") { CHECK_FALSE(deviceLevelIndex({}).has_value()); }

TEST_CASE("deviceLevelIndex: the first handle that is not on a sub-device is chosen")
{
	const std::array<ze_bool_t, 4> flags{1, 1, 0, 0};
	CHECK(deviceLevelIndex(flags) == std::optional<std::size_t>{2});

	const std::array<ze_bool_t, 2> first{0, 1};
	CHECK(deviceLevelIndex(first) == std::optional<std::size_t>{0});
}

TEST_CASE("deviceLevelIndex: with only sub-device handles the first is chosen")
{
	const std::array<ze_bool_t, 3> flags{1, 1, 1};
	CHECK(deviceLevelIndex(flags) == std::optional<std::size_t>{0});
}

TEST_CASE("deviceLevelIndex: a single device-level handle is chosen")
{
	const std::array<ze_bool_t, 1> flags{0};
	CHECK(deviceLevelIndex(flags) == std::optional<std::size_t>{0});
}

// ---------------------------------------------------------------------------
// knownDevState
// ---------------------------------------------------------------------------

TEST_CASE("knownDevState: a successful call with flags gives the flags")
{
	for (const bool computeSkipped : {false, true}) {
		CHECK(knownDevState(ZE_RESULT_SUCCESS, ZES_DEVICE_STATE_EXT_FLAG_NORMAL, computeSkipped) ==
			  std::optional<zes_device_state_ext_flags_t>{ZES_DEVICE_STATE_EXT_FLAG_NORMAL});
		CHECK(knownDevState(ZE_RESULT_SUCCESS,
							ZES_DEVICE_STATE_EXT_FLAG_SURVIVABILITY | ZES_DEVICE_STATE_EXT_FLAG_FLASH_OVERRIDE,
							computeSkipped) ==
			  std::optional<zes_device_state_ext_flags_t>{ZES_DEVICE_STATE_EXT_FLAG_SURVIVABILITY |
														  ZES_DEVICE_STATE_EXT_FLAG_FLASH_OVERRIDE});
	}
}

TEST_CASE("knownDevState: no flags means the state is unknown")
{
	CHECK_FALSE(knownDevState(ZE_RESULT_SUCCESS, 0, false).has_value());
	CHECK_FALSE(knownDevState(ZE_RESULT_SUCCESS, 0, true).has_value());
}

TEST_CASE("knownDevState: a failed call means the state is unknown, whatever it returned")
{
	for (const bool computeSkipped : {false, true}) {
		CHECK_FALSE(knownDevState(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE, 0, computeSkipped).has_value());
		CHECK_FALSE(
			knownDevState(ZE_RESULT_ERROR_UNKNOWN, ZES_DEVICE_STATE_EXT_FLAG_NORMAL, computeSkipped).has_value());
	}
}

TEST_CASE("knownDevState: SURVIVABILITY_MODE_DETECTED means survivability mode for a device the compute runtime skips")
{
	CHECK(knownDevState(ZE_RESULT_ERROR_SURVIVABILITY_MODE_DETECTED, 0, true) ==
		  std::optional<zes_device_state_ext_flags_t>{ZES_DEVICE_STATE_EXT_FLAG_SURVIVABILITY});
}

TEST_CASE("knownDevState: SURVIVABILITY_MODE_DETECTED is unknown for a device the compute runtime reports")
{
	// Some UMD versions return it for healthy devices
	CHECK_FALSE(knownDevState(ZE_RESULT_ERROR_SURVIVABILITY_MODE_DETECTED, 0, false).has_value());
	CHECK_FALSE(
		knownDevState(ZE_RESULT_ERROR_SURVIVABILITY_MODE_DETECTED, ZES_DEVICE_STATE_EXT_FLAG_SURVIVABILITY, false)
			.has_value());
}

// ---------------------------------------------------------------------------
// deviceConditionOf
// ---------------------------------------------------------------------------

TEST_CASE("deviceConditionOf: each state gets its name and recovery action")
{
	struct Expected
	{
		zes_device_state_ext_flags_t flags;
		std::string_view state;
		std::string_view action;
		bool normal;
	};
	for (const Expected &e : {
			 Expected{ZES_DEVICE_STATE_EXT_FLAG_NORMAL, "normal", "None", true},
			 Expected{ZES_DEVICE_STATE_EXT_FLAG_SURVIVABILITY, "survivability mode", "Firmware update and GPU reset",
					  false},
			 Expected{ZES_DEVICE_STATE_EXT_FLAG_WEDGED, "wedged", "Reset", false},
			 Expected{ZES_DEVICE_STATE_EXT_FLAG_GPU_LOST, "GPU lost", "Reboot", false},
			 Expected{ZES_DEVICE_STATE_EXT_FLAG_DRIVER_NOT_LOADED, "driver not loaded", "Load the kernel driver",
					  false},
			 Expected{ZES_DEVICE_STATE_EXT_FLAG_FLASH_OVERRIDE, "flash override", "FDO firmware update and GPU reset",
					  false},
			 Expected{0, "unknown", "None", false},
		 }) {
		const auto condition = deviceConditionOf(e.flags);
		CHECK_MESSAGE(condition.state == e.state, e.flags);
		CHECK_MESSAGE(condition.recoveryAction == e.action, e.flags);
		CHECK_MESSAGE(condition.normal == e.normal, e.flags);
	}
}

TEST_CASE("deviceConditionOf: a device is not normal when another state is set alongside the normal flag")
{
	const auto condition = deviceConditionOf(ZES_DEVICE_STATE_EXT_FLAG_NORMAL | ZES_DEVICE_STATE_EXT_FLAG_WEDGED);
	CHECK(condition.state == "wedged");
	CHECK_FALSE(condition.normal);
}

TEST_CASE("deviceConditionOf: a device in boot survivability mode, as xe reports it, is in survivability mode")
{
	// xe on kernel 7.0 reports a device in boot survivability mode as wedged and in survivability mode (0x6)
	const auto condition =
		deviceConditionOf(ZES_DEVICE_STATE_EXT_FLAG_WEDGED | ZES_DEVICE_STATE_EXT_FLAG_SURVIVABILITY);
	CHECK(condition.state == "survivability mode");
	CHECK(condition.recoveryAction == "Firmware update and GPU reset");
	CHECK_FALSE(condition.normal);
}

TEST_CASE("deviceConditionOf: with several states the one needing the most to recover is shown")
{
	CHECK(deviceConditionOf(ZES_DEVICE_STATE_EXT_FLAG_SURVIVABILITY | ZES_DEVICE_STATE_EXT_FLAG_WEDGED |
							ZES_DEVICE_STATE_EXT_FLAG_FLASH_OVERRIDE)
			  .state == "survivability mode");
	CHECK(deviceConditionOf(ZES_DEVICE_STATE_EXT_FLAG_GPU_LOST | ZES_DEVICE_STATE_EXT_FLAG_WEDGED).state == "GPU lost");
	CHECK(deviceConditionOf(ZES_DEVICE_STATE_EXT_FLAG_DRIVER_NOT_LOADED | ZES_DEVICE_STATE_EXT_FLAG_WEDGED).state ==
		  "driver not loaded");
	CHECK(deviceConditionOf(ZES_DEVICE_STATE_EXT_FLAG_WEDGED | ZES_DEVICE_STATE_EXT_FLAG_FLASH_OVERRIDE).state ==
		  "wedged");
}

// ---------------------------------------------------------------------------
// resetRecoveryActionOf / deviceRecoveryActionOf
// ---------------------------------------------------------------------------

TEST_CASE("resetRecoveryActionOf: none, a repair, or any other reset reason")
{
	CHECK(resetRecoveryActionOf(0) == "None");
	CHECK(resetRecoveryActionOf(ZES_RESET_REASON_FLAG_WEDGED) == "Reset");
	CHECK(resetRecoveryActionOf(ZES_RESET_REASON_FLAG_REPAIR) == "Reset to repair");
	CHECK(resetRecoveryActionOf(ZES_RESET_REASON_FLAG_WEDGED | ZES_RESET_REASON_FLAG_REPAIR) == "Reset to repair");
}

TEST_CASE("deviceRecoveryActionOf: a state that is not normal needs its own action, whatever the reset reasons")
{
	using Action = std::optional<std::string_view>;
	CHECK(deviceRecoveryActionOf(ZES_DEVICE_STATE_EXT_FLAG_SURVIVABILITY, 0) ==
		  Action{"Firmware update and GPU reset"});
	CHECK(deviceRecoveryActionOf(ZES_DEVICE_STATE_EXT_FLAG_WEDGED | ZES_DEVICE_STATE_EXT_FLAG_SURVIVABILITY,
								 std::nullopt) == Action{"Firmware update and GPU reset"});
	CHECK(deviceRecoveryActionOf(ZES_DEVICE_STATE_EXT_FLAG_WEDGED, ZES_RESET_REASON_FLAG_REPAIR) == Action{"Reset"});
}

TEST_CASE("deviceRecoveryActionOf: a normal or unknown state takes the action from the reset reasons")
{
	using Action = std::optional<std::string_view>;
	for (const zes_device_state_ext_flags_t state :
		 {zes_device_state_ext_flags_t{ZES_DEVICE_STATE_EXT_FLAG_NORMAL}, zes_device_state_ext_flags_t{0}}) {
		CHECK_MESSAGE(deviceRecoveryActionOf(state, 0) == Action{"None"}, state);
		CHECK_MESSAGE(deviceRecoveryActionOf(state, ZES_RESET_REASON_FLAG_WEDGED) == Action{"Reset"}, state);
		CHECK_MESSAGE(deviceRecoveryActionOf(state, ZES_RESET_REASON_FLAG_REPAIR) == Action{"Reset to repair"}, state);
	}
}

TEST_CASE("deviceRecoveryActionOf: nothing is known when the state does not tell and the reset reasons cannot be read")
{
	CHECK_FALSE(deviceRecoveryActionOf(ZES_DEVICE_STATE_EXT_FLAG_NORMAL, std::nullopt).has_value());
	CHECK_FALSE(deviceRecoveryActionOf(0, std::nullopt).has_value());
}
