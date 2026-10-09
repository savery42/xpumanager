/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef UTILITY_BITMASK_ENUM_H
#define UTILITY_BITMASK_ENUM_H

#include <type_traits>

/**
 * Bitwise operators for enum classes that opt in.
 *
 * An enum opts in by declaring, next to itself and in its own namespace,
 *
 * @code
 *     consteval bool enableBitmaskOperators(MyEnum) { return true; }
 * @endcode
 *
 * Argument-dependent lookup finds that declaration from here, so no foreign namespace has to be
 * reopened. Unrelated enums never gain the operators.
 *
 * The operators are constrained templates at global scope, which ordinary lookup finds from any
 * namespace that does not declare an operator of the same name itself.
 */
template <typename Enum>
concept BitmaskEnum = std::is_enum_v<Enum> && requires { requires enableBitmaskOperators(Enum{}); };

template <BitmaskEnum Enum> [[nodiscard]] constexpr Enum operator|(Enum a, Enum b) noexcept
{
	using U = std::underlying_type_t<Enum>;
	// NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange): BitmaskEnum types are bitmasks.
	return static_cast<Enum>(static_cast<U>(a) | static_cast<U>(b));
}

template <BitmaskEnum Enum> [[nodiscard]] constexpr Enum operator&(Enum a, Enum b) noexcept
{
	using U = std::underlying_type_t<Enum>;
	// NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange): BitmaskEnum types are bitmasks.
	return static_cast<Enum>(static_cast<U>(a) & static_cast<U>(b));
}

template <BitmaskEnum Enum> [[nodiscard]] constexpr Enum operator~(Enum a) noexcept
{
	using U = std::underlying_type_t<Enum>;
	// NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange): BitmaskEnum types are bitmasks.
	return static_cast<Enum>(static_cast<U>(~static_cast<U>(a)));
}

template <BitmaskEnum Enum> constexpr Enum &operator|=(Enum &a, Enum b) noexcept { return a = a | b; }

/// True when @p field has at least one bit of @p mask set.
template <BitmaskEnum Enum> [[nodiscard]] constexpr bool hasAny(Enum field, Enum mask) noexcept
{
	return (field & mask) != Enum{};
}

/// True when @p field has every bit of @p mask set; trivially true for an empty @p mask.
template <BitmaskEnum Enum> [[nodiscard]] constexpr bool hasAll(Enum field, Enum mask) noexcept
{
	return (field & mask) == mask;
}

#endif // UTILITY_BITMASK_ENUM_H
