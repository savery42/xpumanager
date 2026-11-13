/*
 * Copyright (C) 2025-2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef _CMDS_H
#define _CMDS_H

#include <cstdarg>
#include <device.h>
#include <driver.h>
#include <list>
#include "requirements.h"
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern std::string progName;

enum GAP
{
	TITLE = 0,
	BLANK = 0,
	HEADING = 2,
	SUB_HEADING = 30,
	SUB_HEADING2 = 37,
};

enum HELP
{
	SHORT_HELP,
	FULL_HELP,
};

// The constructors are defined in cmds.cpp: inlined into std::vector::emplace_back, GCC 13
// misjudges the size of the element being built and fails the build with -Warray-bounds or
// -Wstringop-overflow.
struct helpCmd
{
	char line[MAX_PATH]{};
	int char_gap{};

	helpCmd();
	helpCmd(GAP gap, const char *fmt, ...);
	helpCmd(GAP gap);
};

struct arg_struct
{
	int argc;
	char **argv;
	driver sm;						   // sm is short for sysman
	Capability held{Capability::NONE}; // resolved once in main(), before driver init
};

/// argv tokens after the subcommand name, i.e. argv[2..argc).
[[nodiscard]] inline std::span<char *const> subcommandArgs(const arg_struct &args)
{
	if (args.argc < 2) {
		return {};
	}
	return std::span<char *const>{args.argv, static_cast<std::size_t>(args.argc)}.subspan(2);
}

class cmds
{
protected:
	std::string name;

public:
	cmds() {};
	const std::string &getName() { return name; }
	virtual ~cmds() {};
	void printHelp(std::vector<helpCmd> helpList, HELP helpType = FULL_HELP);
	virtual void help(HELP helpType = FULL_HELP) = 0;
	virtual int run(arg_struct *args) = 0;
	/**
	 * @brief What the dispatcher must check and bring up before run().
	 *
	 * Non-virtual so that -h/--help never needs privilege or the driver.
	 */
	[[nodiscard]] CommandPlan plan(std::span<char *const> cmdArgs) const;
	/// Whether run() can still act when the driver fails to initialize. Default: no.
	[[nodiscard]] virtual bool recoversWithoutDriver(std::span<char *const> cmdArgs) const;

protected:
	/// Override to declare what this command needs. Default: sysman only, no privilege.
	[[nodiscard]] virtual CommandPlan needs(std::span<char *const> cmdArgs) const;
};

namespace CLI {
class App;
} // namespace CLI

/// Parses @p cmdArgs with @p app without printing; false on a parse error or help request.
[[nodiscard]] bool parsesCleanly(CLI::App &app, std::span<char *const> cmdArgs);

typedef void (cmds::*helpFunc)(HELP helpType);
typedef int (cmds::*runFunc)();

struct cmd_struct
{
	helpFunc hf;
	runFunc rf;
};

/// Return the "DDDD:BB:DD.F" PCI address for a device.
/// Used by IAL callers to pass a plain BDF to OAL fdinfo functions.
inline std::string devPciAddr(const devInfo &dev)
{
	if (dev.dev == nullptr) {
		return {};
	}
	return dev.dev->getBDFStr();
}

/**
 * @brief PCI addresses of a list of devices.
 *
 * @param[in] devList  Devices to collect addresses from.
 * @return devPciAddr() of every device in @p devList, skipping devices that have none.
 */
inline std::unordered_set<std::string> devPciAddrs(const std::vector<devInfo> &devList)
{
	std::unordered_set<std::string> addrs;
	for (const auto &dev : devList) {
		if (auto addr = devPciAddr(dev); !addr.empty()) {
			addrs.insert(std::move(addr));
		}
	}
	return addrs;
}

#endif
