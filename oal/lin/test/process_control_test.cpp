/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 * Tests for sendStopSignal() against a forked child and against pids it must refuse.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "process_control.h"

#include <csignal>
#include <sys/wait.h>
#include <unistd.h>

TEST_CASE("sendStopSignal rejects pids that would signal a group or overflow")
{
	CHECK(sendStopSignal(0, StopSignal::Terminate) == SignalResult::Failed);
	CHECK(sendStopSignal(0x80000000U, StopSignal::Terminate) == SignalResult::Failed);
}

TEST_CASE("sendStopSignal terminates a child, then reports it gone")
{
	int ready[2] = {-1, -1};
	REQUIRE(pipe(ready) == 0);
	const pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		// doctest installs a SIGTERM handler that reports a crash; the child must die by the signal.
		signal(SIGTERM, SIG_DFL);
		const char go = 'g';
		if (write(ready[1], &go, 1) != 1) {
			_exit(1);
		}
		for (;;) {
			pause();
		}
	}
	char go = 0;
	REQUIRE(read(ready[0], &go, 1) == 1);
	close(ready[0]);
	close(ready[1]);

	CHECK(sendStopSignal(static_cast<std::uint32_t>(child), StopSignal::Terminate) == SignalResult::Sent);

	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFSIGNALED(status));
	CHECK(WTERMSIG(status) == SIGTERM);

	CHECK(sendStopSignal(static_cast<std::uint32_t>(child), StopSignal::Kill) == SignalResult::NoSuchProcess);
}

TEST_CASE("sendStopSignal reports a process the caller may not signal")
{
	if (geteuid() == 0) {
		MESSAGE("running as root, init is signalable; skipped");
		return;
	}
	CHECK(sendStopSignal(1, StopSignal::Terminate) == SignalResult::NotPermitted);
}
