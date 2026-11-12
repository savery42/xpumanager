/*
 * Copyright (C) 2025-2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "device.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace xpum::detail {

/**
 * @brief Parse a decimal uint32 from a string_view.
 *
 * Uses std::from_chars — locale-independent and non-throwing.
 *
 * @param[in] sv The string_view to parse.
 * @return The parsed value, or std::nullopt if @p sv is not a complete
 *         decimal representation of a value in [0, UINT32_MAX].
 */
inline std::optional<uint32_t> parseUint32(std::string_view sv) noexcept
{
	uint32_t val{};
	// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	// from_chars takes raw pointer pairs — no alternative API exists.
	const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), val);
	// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	if (ec != std::errc{} || ptr != sv.data() + sv.size()) {
		return std::nullopt;
	}
	return val;
}

/**
 * @brief Remove duplicate entries from @p devList, keeping the first occurrence of each index.
 *
 * Preserves insertion order. Used after resolving comma-separated device tokens to
 * eliminate duplicates arising from repeated indices (e.g. "0,0").
 *
 * @param[in,out] devList The device list to deduplicate in place.
 */
inline void deduplicateByIndex(std::vector<devInfo> &devList)
{
	std::unordered_set<uint32_t> seen;
	std::erase_if(devList, [&](const devInfo &d) { return !seen.insert(d.index).second; });
}

/**
 * @brief Pick the handle that describes the device as a whole.
 *
 * A device lists its scheduler or standby domains once for itself and, on a multi-tile device,
 * again for each tile. The first one that is not on a sub-device is the device's own; a device
 * that lists only sub-device domains is described by the first of them.
 *
 * @param[in] onSubdevice One flag per handle, in handle order: nonzero when the handle is on a sub-device.
 *
 * @retval index   Position of the first handle that is not on a sub-device, else 0.
 * @retval nullopt There are no handles.
 */
[[nodiscard]] inline std::optional<std::size_t> deviceLevelIndex(std::span<const ze_bool_t> onSubdevice) noexcept
{
	const auto device = std::ranges::find(onSubdevice, ze_bool_t{0});
	if (device != onSubdevice.end()) {
		return static_cast<std::size_t>(device - onSubdevice.begin());
	}
	if (!onSubdevice.empty()) {
		return std::size_t{0};
	}
	return std::nullopt;
}

/**
 * @brief Reads the result of zesDeviceGetState() with a zes_device_ext_state_t chained.
 *
 * A failed call says nothing about the state, with one exception. Some UMD versions return
 * ZE_RESULT_ERROR_SURVIVABILITY_MODE_DETECTED for healthy devices, so that result is taken as survivability
 * mode only when the compute runtime came up without reporting the device, as it does for a device in
 * survivability mode. Flags of 0 mean the driver cannot determine the state.
 *
 * @param[in] result         Result of zesDeviceGetState().
 * @param[in] flags          zes_device_ext_state_t::flags it returned, ZES_DEVICE_STATE_EXT_FLAG_* values.
 * @param[in] computeSkipped The compute runtime initialized and does not report the device.
 *
 * @return The flags, or std::nullopt when the state is unknown: the call failed or it returned no flags.
 */
[[nodiscard]] constexpr std::optional<zes_device_state_ext_flags_t>
knownDevState(ze_result_t result, zes_device_state_ext_flags_t flags, bool computeSkipped) noexcept
{
	if (result == ZE_RESULT_ERROR_SURVIVABILITY_MODE_DETECTED && computeSkipped) {
		return ZES_DEVICE_STATE_EXT_FLAG_SURVIVABILITY;
	}
	if (result != ZE_RESULT_SUCCESS || flags == 0) {
		return std::nullopt;
	}
	return flags;
}

/** @brief A device state as shown to the user, with what the device needs to recover from it. */
struct DeviceCondition
{
	std::string_view state;			 ///< e.g. "normal" or "survivability mode"
	std::string_view recoveryAction; ///< e.g. "None" or "Firmware update and GPU reset"
	bool normal;					 ///< the device is operating normally
};

/**
 * @brief Names the state the device state extension reports, and what the device needs to recover.
 *
 * Several flags can be set at once; the state that needs the most to recover from is shown:
 * survivability mode, then a lost GPU, a missing kernel driver, a wedged device and a flash override.
 *
 * @param[in] flags ZES_DEVICE_STATE_EXT_FLAG_* values of the device.
 *
 * @return The state and its recovery action. A device is normal only when ZES_DEVICE_STATE_EXT_FLAG_NORMAL
 *         is set and no other state applies; flags of 0 are an "unknown" state.
 */
[[nodiscard]] constexpr DeviceCondition deviceConditionOf(zes_device_state_ext_flags_t flags) noexcept
{
	if ((flags & ZES_DEVICE_STATE_EXT_FLAG_SURVIVABILITY) != 0U) {
		return {"survivability mode", "Firmware update and GPU reset", false};
	}
	if ((flags & ZES_DEVICE_STATE_EXT_FLAG_GPU_LOST) != 0U) {
		return {"GPU lost", "Reboot", false};
	}
	if ((flags & ZES_DEVICE_STATE_EXT_FLAG_DRIVER_NOT_LOADED) != 0U) {
		return {"driver not loaded", "Load the kernel driver", false};
	}
	if ((flags & ZES_DEVICE_STATE_EXT_FLAG_WEDGED) != 0U) {
		return {"wedged", "Reset", false};
	}
	if ((flags & ZES_DEVICE_STATE_EXT_FLAG_FLASH_OVERRIDE) != 0U) {
		return {"flash override", "FDO firmware update and GPU reset", false};
	}
	if ((flags & ZES_DEVICE_STATE_EXT_FLAG_NORMAL) != 0U) {
		return {"normal", "None", true};
	}
	return {"unknown", "None", false};
}

/**
 * @brief Says what the reset reasons the driver reports call for.
 *
 * @param[in] reasons Reset reason flags from the device state.
 *
 * @return "None" when no reset is needed, "Reset to repair" when a repair is waiting on one,
 *         and "Reset" for any other reason, such as a wedged device.
 */
[[nodiscard]] constexpr std::string_view resetRecoveryActionOf(zes_reset_reason_flags_t reasons) noexcept
{
	if (reasons == 0) {
		return "None";
	}
	if ((reasons & ZES_RESET_REASON_FLAG_REPAIR) != 0U) {
		return "Reset to repair";
	}
	return "Reset";
}

/**
 * @brief Says what a device needs to keep working, from its state and its reset reasons.
 *
 * A known state that is not normal, e.g. survivability mode, needs the action deviceConditionOf() gives for
 * it. Otherwise, for a normal or an unknown state, the reset reasons decide, see resetRecoveryActionOf().
 *
 * @param[in] state   ZES_DEVICE_STATE_EXT_FLAG_* values of the device, 0 when unknown.
 * @param[in] reasons Reset reason flags of the device, std::nullopt when they cannot be read.
 *
 * @return The action, or std::nullopt when the state is normal or unknown and the reset reasons cannot be read.
 */
[[nodiscard]] constexpr std::optional<std::string_view>
deviceRecoveryActionOf(zes_device_state_ext_flags_t state, std::optional<zes_reset_reason_flags_t> reasons) noexcept
{
	if (state != 0) {
		const DeviceCondition condition = deviceConditionOf(state);
		if (!condition.normal) {
			return condition.recoveryAction;
		}
	}
	if (reasons.has_value()) {
		return resetRecoveryActionOf(*reasons);
	}
	return std::nullopt;
}

/**
 * @brief Says what @p dev needs to keep working, from its state and the reset reasons the driver reports.
 *
 * @param[in] dev The device.
 *
 * @return See deviceRecoveryActionOf(zes_device_state_ext_flags_t, std::optional<zes_reset_reason_flags_t>).
 */
[[nodiscard]] inline std::optional<std::string_view> deviceRecoveryActionOf(device &dev)
{
	zes_reset_reason_flags_t reasons = 0;
	const bool known = dev.getResetReasons(reasons) == ZE_RESULT_SUCCESS;
	return deviceRecoveryActionOf(dev.getDevState(), known ? std::optional{reasons} : std::nullopt);
}

} // namespace xpum::detail
