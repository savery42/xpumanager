/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "requirements.h"

#include <algorithm>
#include <limits>
#include <ranges>
#include <type_traits>

Resolution resolve(const CommandPlan &plan, Capability held)
{
	Resolution r{.missing = missing(held, plan.required), .driver = plan.driver};
	for (const OptionalFeature &f : plan.optional) {
		if (hasAll(held, f.capability)) {
			r.driver = std::max(r.driver, f.driver);
		} else {
			r.skipped.push_back(f);
		}
	}
	return r;
}

std::string describeAll(Capability set)
{
	using Bits = std::underlying_type_t<Capability>;
	const auto single = [](unsigned bit) { return static_cast<Capability>(static_cast<Bits>(1U << bit)); };
	std::string out;
	for (const Capability c : std::views::iota(0U, unsigned{std::numeric_limits<Bits>::digits}) |
								  std::views::transform(single) |
								  std::views::filter([set](Capability c) { return hasAll(set, c); })) {
		out += out.empty() ? "" : "; and ";
		out += describe(c);
	}
	return out;
}
