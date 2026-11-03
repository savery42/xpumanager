/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 * Tests for keyboard::decodeKey(). It is constexpr and header-only, so no
 * terminal is needed.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "keyboard.h"

#include <array>

static_assert(keyboard::decodeKey(std::array<unsigned char, 3>{0x1b, '[', 'A'}) == keyboard::KEY_UP);
static_assert(keyboard::decodeKey(std::array<unsigned char, 1>{'q'}) == 'q');

namespace {

template <std::size_t N> int decode(const std::array<unsigned char, N> &bytes) { return keyboard::decodeKey(bytes); }

} // namespace

TEST_CASE("decodeKey plain bytes")
{
	CHECK(keyboard::decodeKey({}) == keyboard::NO_KEY);
	CHECK(decode<1>({'q'}) == 'q');
	CHECK(decode<1>({0x03}) == 0x03);
	CHECK(decode<1>({0x1b}) == keyboard::KEY_ESCAPE);
}

TEST_CASE("decodeKey escape sequences")
{
	CHECK(decode<3>({0x1b, '[', 'A'}) == keyboard::KEY_UP);
	CHECK(decode<3>({0x1b, '[', 'B'}) == keyboard::KEY_DOWN);
	CHECK(decode<3>({0x1b, '[', 'H'}) == keyboard::KEY_HOME);
	CHECK(decode<3>({0x1b, '[', 'F'}) == keyboard::KEY_END);
	CHECK(decode<4>({0x1b, '[', '5', '~'}) == keyboard::KEY_PAGE_UP);
	CHECK(decode<4>({0x1b, '[', '6', '~'}) == keyboard::KEY_PAGE_DOWN);
}

TEST_CASE("decodeKey rejects what it does not know")
{
	CHECK(decode<3>({0x1b, '[', '5'}) == keyboard::NO_KEY);
	CHECK(decode<3>({0x1b, '[', 'Z'}) == keyboard::NO_KEY);
	CHECK(decode<3>({0x1b, 'O', 'A'}) == keyboard::NO_KEY);
	CHECK(decode<2>({0x1b, '['}) == keyboard::NO_KEY);
}
