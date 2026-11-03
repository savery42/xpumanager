/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef OAL_KEYBOARD_H
#define OAL_KEYBOARD_H

#include <chrono>
#include <memory>
#include <optional>
#include <span>

namespace keyboard {

inline constexpr int NO_KEY = -1;
inline constexpr int KEY_ESCAPE = 0x1b;
inline constexpr int KEY_UP = 0x100;
inline constexpr int KEY_DOWN = 0x101;
inline constexpr int KEY_PAGE_UP = 0x102;
inline constexpr int KEY_PAGE_DOWN = 0x103;
inline constexpr int KEY_HOME = 0x104;
inline constexpr int KEY_END = 0x105;

/**
 * @brief Decodes the bytes of one read from a raw-mode terminal into a key.
 *
 * @param[in] bytes  What one read() returned.
 *
 * @retval NO_KEY      @p bytes is empty, or an escape sequence this does not know.
 * @retval KEY_ESCAPE  A lone ESC byte.
 * @retval KEY_*       A recognised cursor/paging sequence (CSI A/B/H/F, CSI 5~/6~).
 * @retval byte        Otherwise, the first byte.
 */
[[nodiscard]] constexpr int decodeKey(std::span<const unsigned char> bytes) noexcept
{
	if (bytes.empty()) {
		return NO_KEY;
	}
	if (bytes[0] != KEY_ESCAPE) {
		return bytes[0];
	}
	if (bytes.size() == 1) {
		return KEY_ESCAPE;
	}
	if (bytes.size() < 3 || bytes[1] != '[') {
		return NO_KEY;
	}
	const bool tilde = bytes.size() >= 4 && bytes[3] == '~';
	switch (bytes[2]) {
	case 'A':
		return KEY_UP;
	case 'B':
		return KEY_DOWN;
	case 'H':
		return KEY_HOME;
	case 'F':
		return KEY_END;
	case '5':
		return tilde ? KEY_PAGE_UP : NO_KEY;
	case '6':
		return tilde ? KEY_PAGE_DOWN : NO_KEY;
	default:
		return NO_KEY;
	}
}

/**
 * @brief Puts stdin in raw mode for single-key input while it lives.
 *
 * A no-op when stdin is not a terminal, and on Windows. Raw mode also turns off
 * ISIG, so Ctrl-C arrives as the byte 0x03 instead of SIGINT.
 */
class RawKeyboard
{
public:
	RawKeyboard();
	~RawKeyboard();
	RawKeyboard(const RawKeyboard &) = delete;
	RawKeyboard &operator=(const RawKeyboard &) = delete;
	RawKeyboard(RawKeyboard &&) = delete;
	RawKeyboard &operator=(RawKeyboard &&) = delete;

	/// @retval true  stdin is in raw mode and will be restored on destruction.
	[[nodiscard]] bool active() const noexcept { return saved != nullptr; }

private:
	struct Saved;
	std::unique_ptr<Saved> saved;
};

/**
 * @brief Waits for one key on stdin.
 *
 * @param[in] timeout  How long to wait; std::nullopt waits indefinitely.
 *
 * @retval NO_KEY  Timed out, read failed, or the platform has no key input.
 * @retval key     As decodeKey() returns it.
 */
[[nodiscard]] int readKey(std::optional<std::chrono::milliseconds> timeout) noexcept;

} // namespace keyboard

#endif // OAL_KEYBOARD_H
