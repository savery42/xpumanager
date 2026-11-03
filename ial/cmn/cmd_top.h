/*
 * Copyright (C) 2025-2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef _CMD_TOP_H
#define _CMD_TOP_H

#include "cmds.h"
#include <span>

class cmdTop : public cmds
{
public:
	cmdTop() { name = "top"; }
	~cmdTop() override = default;
	cmdTop(const cmdTop &) = delete;
	cmdTop &operator=(const cmdTop &) = delete;
	cmdTop(cmdTop &&) = delete;
	cmdTop &operator=(cmdTop &&) = delete;
	void help(HELP helpType = FULL_HELP) override;
	int run(arg_struct *args) override;

protected:
	[[nodiscard]] CommandPlan needs(std::span<char *const> cmdArgs) const override;
};

#endif
