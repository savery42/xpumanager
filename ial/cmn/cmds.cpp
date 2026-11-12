/*
 * Copyright (C) 2025-2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 */

#include "cmds.h"
#include "driver_util.h"
#include <algorithm>
#include <ranges>
#include <string_view>
#include <tuple>

/**
 * @brief Prints formatted help information for command line interfaces
 *
 * This function displays help content in different formats based on the help type.
 * For SHORT_HELP, it shows a condensed single-line description. For FULL_HELP,
 * it displays comprehensive usage information with proper formatting and spacing.
 *
 * @param helpList Vector of help command structures containing formatted help text
 * @param helpType Type of help to display (SHORT_HELP for brief, FULL_HELP for detailed)
 */
void cmds::printHelp(std::vector<helpCmd> helpList, HELP helpType)
{
	if (helpType == SHORT_HELP) {
		/* Just print the first line of each subcommand's help because it contains the description */
		if (!helpList.empty()) {
			PRINT("  {0:<{1}}{2}\n", name, 28, helpList[0].line);
		}
	} else {
		for (const auto &it2 : helpList) {
			PRINT("{0:<{1}}{2}\n", "", it2.char_gap, it2.line);
		}
	}
}

namespace {

// Every subcommand registers "-h,--help" with CLI11 and finishes parsing before
// it touches a device, so a help request never needs the driver.
[[nodiscard]] bool requestsHelp(std::span<char *const> cmdArgs)
{
	return std::ranges::any_of(cmdArgs, [](std::string_view token) { return token == "-h" || token == "--help"; });
}

} // namespace

/**
 * @brief Reports how much of Level Zero the dispatcher must initialize before run().
 *
 * @param[in] cmdArgs  argv tokens after the subcommand name.
 *
 * @retval DriverMode::None     @p cmdArgs contains -h or --help.
 * @retval driverMode(cmdArgs)  Otherwise.
 */
DriverMode cmds::requiredDriver(std::span<char *const> cmdArgs) const
{
	return requestsHelp(cmdArgs) ? DriverMode::None : driverMode(cmdArgs);
}

/**
 * @brief Reports how much of Level Zero this command needs for @p cmdArgs.
 *
 * Only called when @p cmdArgs has no -h/--help. A command that prints help for
 * other forms, such as a bare invocation, returns DriverMode::None for them.
 *
 * @param[in] cmdArgs  argv tokens after the subcommand name.
 *
 * @retval DriverMode::Full  Default for every command that queries devices.
 */
DriverMode cmds::driverMode(UNUSED std::span<char *const> cmdArgs) const { return DriverMode::Full; }

/**
 * @brief Reports whether run() can still act on @p cmdArgs when the driver fails to initialize.
 *
 * A command that returns true reaches its devices some other way, e.g. through
 * PCI sysfs with driver::findDeviceForRecovery(), so a node whose GPUs Level
 * Zero cannot bring up can still be recovered.
 *
 * @param[in] cmdArgs  argv tokens after the subcommand name.
 *
 * @retval false  Default: a driver init failure ends the command.
 */
bool cmds::recoversWithoutDriver(UNUSED std::span<char *const> cmdArgs) const { return false; }

/**
 * @brief Resolves a --device value against both the GPUs Level Zero reports and those listed for their state.
 *
 * Each comma-separated selector is looked up among the GPUs listed for their state (e.g. survivability
 * mode) first, so one that names such a GPU is not also reported as an unknown device. A selector that
 * names no GPU is reported and skipped.
 *
 * @param[in] sm   The driver that enumerated the GPUs.
 * @param[in] spec Device indices or PCI addresses, comma-separated; empty selects every GPU.
 *
 * @return The selected GPUs of each kind, without duplicates, in the order they were named.
 */
SelectedDevices selectDevices(driver &sm, std::string_view spec)
{
	SelectedDevices selected;
	if (spec.empty()) {
		std::ignore = sm.findDevice("", &selected.normal);
		sm.findSurvDevice("", &selected.survivability);
		return selected;
	}
	for (const auto part : spec | std::views::split(',')) {
		const std::string token{part.begin(), part.end()};
		if (token.empty()) {
			continue;
		}
		const std::size_t before = selected.survivability.size();
		sm.findSurvDevice(token.c_str(), &selected.survivability);
		if (selected.survivability.size() == before) {
			// A selector that matches nothing is logged by findDevice.
			std::ignore = sm.findDevice(token.c_str(), &selected.normal);
		}
	}
	xpum::detail::deduplicateByIndex(selected.normal);
	xpum::detail::deduplicateByIndex(selected.survivability);
	return selected;
}
