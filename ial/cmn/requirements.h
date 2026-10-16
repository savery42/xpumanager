/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef IAL_REQUIREMENTS_H
#define IAL_REQUIREMENTS_H

#include <privilege.h>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/// How much of Level Zero must be initialized before a command runs. Ordered: a larger value includes the smaller.
enum class DriverMode
{
	None,		// args->sm is never touched
	SysmanOnly, // zesInit only; the compute runtime (zeInit) is never loaded
	Full,		// zesInit and zeInit
};

/**
 * @brief A command feature that is skipped when its privilege is not held.
 *
 * Declaring the driver level next to the capability keeps the two in one
 * place: the dispatcher brings up @c driver only while the feature is available.
 */
struct OptionalFeature
{
	std::string_view name;				 ///< User-facing, e.g. "EU active/stall columns"
	Capability capability;				 ///< What the feature needs, all of it
	DriverMode driver{DriverMode::None}; ///< Driver level the feature needs while it runs
	bool announce{true};				 ///< Notice when skipped; false where a fallback still serves the feature
};

/**
 * @brief What one invocation needs, decided from its arguments before any driver call.
 */
struct CommandPlan
{
	DriverMode driver{DriverMode::SysmanOnly}; ///< Baseline, before any optional feature
	Capability required{Capability::NONE};	   ///< Missing any of these: refuse before driver init
	/// Skipped with a notice when not held. Must view static storage: a plan outlives the call that made it.
	std::span<const OptionalFeature> optional{};
};

/**
 * @brief A plan checked against what the user holds. Pure data: the caller reports it.
 */
struct Resolution
{
	Capability missing{Capability::NONE};	///< Required but not held
	DriverMode driver{DriverMode::None};	///< Baseline raised by every available optional feature
	std::vector<OptionalFeature> skipped{}; ///< Optional features not held, in declaration order

	[[nodiscard]] bool permitted() const noexcept { return missing == Capability::NONE; }
};

/**
 * @brief Checks @p plan against @p held.
 *
 * @param[in] plan  What the invocation needs.
 * @param[in] held  What the user holds.
 * @return What is missing, the driver level to bring up, and what is skipped.
 */
[[nodiscard]] Resolution resolve(const CommandPlan &plan, Capability held);

/**
 * @brief How to obtain every capability in @p set, joined for a diagnostic message.
 *
 * @param[in] set  Capabilities to describe, e.g. Resolution::missing.
 * @retval "a; and b"  One phrase per capability, lowest bit first.
 * @retval ""          @p set is empty.
 */
[[nodiscard]] std::string describeAll(Capability set);

#endif // IAL_REQUIREMENTS_H
