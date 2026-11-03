/*
 * Copyright (C) 2025-2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 * Minimal ANSI escape helpers. Header-only and portable: Windows enables
 * virtual-terminal processing in stdoutIsTerminal().
 */

#pragma once

#include <cstdio>

namespace term {

// gotoPos is 0-based (row=0, col=0 is the top-left corner).
inline void gotoPos(int row, int col) noexcept { std::fprintf(stdout, "\033[%d;%dH", row + 1, col + 1); }

inline void home() noexcept { std::fputs("\033[H", stdout); }
inline void clearToEol() noexcept { std::fputs("\033[K", stdout); }
inline void clearToEnd() noexcept { std::fputs("\033[J", stdout); }
inline void flushOut() noexcept { std::fflush(stdout); }

/// Draws on the alternate screen with the cursor hidden while it lives.
class AltScreen
{
public:
	AltScreen()
	{
		std::fputs("\033[?1049h\033[?25l\033[H\033[J", stdout);
		std::fflush(stdout);
	}
	~AltScreen()
	{
		std::fputs("\033[0m\033[?25h\033[?1049l", stdout);
		std::fflush(stdout);
	}
	AltScreen(const AltScreen &) = delete;
	AltScreen &operator=(const AltScreen &) = delete;
	AltScreen(AltScreen &&) = delete;
	AltScreen &operator=(AltScreen &&) = delete;
};

// Foreground / background color indices (ANSI 8-color)
inline constexpr int FG_BLACK = 30;
inline constexpr int FG_RED = 31;
inline constexpr int FG_GREEN = 32;
inline constexpr int FG_YELLOW = 33;
inline constexpr int FG_BLUE = 34;
inline constexpr int FG_MAGENTA = 35;
inline constexpr int FG_CYAN = 36;
inline constexpr int FG_WHITE = 37;
inline constexpr int FG_DEFAULT = 39;
inline constexpr int BG_BLACK = 40;
inline constexpr int BG_WHITE = 47;
inline constexpr int BG_DEFAULT = 49;

inline void setFg(int fg) noexcept { std::fprintf(stdout, "\033[%dm", fg); }
inline void setBg(int bg) noexcept { std::fprintf(stdout, "\033[%dm", bg); }
inline void attrReset() noexcept { std::fputs("\033[0m", stdout); }
inline void attrBold() noexcept { std::fputs("\033[1m", stdout); }
inline void attrDim() noexcept { std::fputs("\033[2m", stdout); }
inline void attrUL() noexcept { std::fputs("\033[4m", stdout); }
inline void attrReverse() noexcept { std::fputs("\033[7m", stdout); }

} // namespace term
