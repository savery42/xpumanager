/*
 * Copyright (C) 2025-2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 */

#include "cmds.h"
#include <CLI/CLI.hpp>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <string_view>

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
helpCmd::helpCmd() = default;

helpCmd::helpCmd(GAP gap, const char *fmt, ...) : char_gap(static_cast<int>(gap))
{
	va_list args;
	va_start(args, fmt);
	vsnprintf(line, sizeof(line), fmt, args);
	va_end(args);
}

helpCmd::helpCmd(GAP gap) : char_gap(static_cast<int>(gap)) {}

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
 * @brief Reports what the dispatcher must check and bring up before run().
 *
 * @param[in] cmdArgs  argv tokens after the subcommand name.
 *
 * @retval {.driver = DriverMode::None}  @p cmdArgs contains -h or --help.
 * @retval needs(cmdArgs)                Otherwise.
 */
CommandPlan cmds::plan(std::span<char *const> cmdArgs) const
{
	return requestsHelp(cmdArgs) ? CommandPlan{.driver = DriverMode::None} : needs(cmdArgs);
}

/**
 * @brief Reports what this command needs for @p cmdArgs.
 *
 * Only called when @p cmdArgs has no -h/--help. A command that prints help for
 * other forms, such as a bare invocation, returns DriverMode::None for them.
 *
 * @param[in] cmdArgs  argv tokens after the subcommand name.
 *
 * @retval {.driver = DriverMode::SysmanOnly}  Default: sysman serves every device query except
 *                                             the few that override this.
 */
CommandPlan cmds::needs(UNUSED std::span<char *const> cmdArgs) const { return {}; }

/**
 * @brief Parses @p cmdArgs with @p app, printing nothing.
 *
 * Lets needs() read the options a command declares for run() from the same
 * CLI11 declaration, so the two cannot disagree about a flag's spelling.
 *
 * @param[in,out] app      Options declared exactly as run() declares them.
 * @param[in]     cmdArgs  argv tokens after the subcommand name.
 *
 * @retval true   The arguments parsed; @p app's bound variables hold them.
 * @retval false  A parse error or a help request, both of which run() reports
 *                before it touches a device.
 */
bool parsesCleanly(CLI::App &app, std::span<char *const> cmdArgs)
{
	// CLI11's vector form takes the arguments in reverse order.
	std::vector<std::string> reversed(cmdArgs.rbegin(), cmdArgs.rend());
	try {
		app.parse(reversed);
	} catch (const CLI::Error &) {
		return false;
	}
	return true;
}

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
