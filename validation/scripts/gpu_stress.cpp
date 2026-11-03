// gpu_stress.cpp - headless OpenCL load generator for any GPU with an OpenCL driver
// Build: g++ -std=c++20 -O2 -Wall -Wextra -Wconversion -DCL_TARGET_OPENCL_VERSION=300
//            -o gpu_stress gpu_stress.cpp -lOpenCL -pthread
//
// Generates a chosen kind of load - ALU, memory read/write, copy engine, host<->device bus -
// on one GPU, several, or all of them, at full load or a duty cycle. Meant for thermal and
// fan calibration and for checking that monitoring tools report what is really happening.
//
// Usage:
//   gpu_stress [--list] [--device N[,M...]|all] [--mode MODE] [--duty PCT] [--ramp SEC]
//              [--period-ms N] [--duration SEC] [--buffer-mib N] [--iters N] [--work-items N]
//              [--local-size N] [--batch N] [--no-calibrate] [--timeout SEC] [--quiet]
//   Run with --help for what each option and mode does.
//
// Each device gets its own thread, context and two command queues used in alternation, so the
// CPU submits the next batch while the GPU finishes the previous one and the GPU never idles.
// Dispatches are calibrated to a few tens of milliseconds so status, duty cycles and Ctrl-C
// all respond promptly.

#include <CL/cl.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

// Signal handling
std::atomic<bool> g_stop{false}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables,readability-identifier-naming)
std::mutex g_print; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables,readability-identifier-naming)

void onSignal([[maybe_unused]] int /*sig*/)
{
    g_stop.store(true, std::memory_order_release);
}

// All kernels are float4 streams or a register-only multiply-add loop.
constexpr std::string_view KERNEL_SRC = R"cl(
__kernel void stress(__global float *out, int iters, float m, float c)
{
    int gid = get_global_id(0);
    float base = (float)(gid & 0xFFFFF) * 9.5367431640625e-7f;  // gid / 2^20
    float a0 = base + 0.0f, a1 = base + 0.1f, a2 = base + 0.2f, a3 = base + 0.3f;
    float a4 = base + 0.4f, a5 = base + 0.5f, a6 = base + 0.6f, a7 = base + 0.7f;
    for (int i = 0; i < iters; ++i) {
        a0 = mad(a0, m, c); a1 = mad(a1, m, c); a2 = mad(a2, m, c); a3 = mad(a3, m, c);
        a4 = mad(a4, m, c); a5 = mad(a5, m, c); a6 = mad(a6, m, c); a7 = mad(a7, m, c);
    }
    out[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

// Helper names avoid OpenCL built-ins (mix() is one): OpenCL C has no overloading, so a strict
// compiler rejects a second mix().
// Memory streams use pseudo-random data: GPUs with lossless memory compression move far fewer
// bytes than requested for zeros or other regular data, which would inflate every figure.
uint hash32(uint x)
{
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

// Values in [0, 1), so read_mem's negative sentinel can never match.
float4 pattern(uint i, uint seed)
{
    const uint b = (i * 4u) ^ seed;
    const float scale = 1.0f / 16777216.0f;
    return (float4)((float)(hash32(b) >> 8), (float)(hash32(b + 1u) >> 8), (float)(hash32(b + 2u) >> 8),
                    (float)(hash32(b + 3u) >> 8)) * scale;
}

__kernel void fill_noise(__global float4 *out, uint seed)
{
    size_t gid = get_global_id(0);
    out[gid] = pattern((uint)gid, seed);
}

// The store never happens, but the compiler cannot know that and must keep every load.
__kernel void read_mem(__global const float4 *in, __global float *sink, float never)
{
    size_t gid = get_global_id(0);
    float4 v = in[gid];
    if (v.x == never) { sink[0] = v.y + v.z + v.w; }
}

__kernel void write_mem(__global float4 *out, uint seed)
{
    size_t gid = get_global_id(0);
    out[gid] = pattern((uint)gid, seed);
}

__kernel void triad(__global float4 *a, __global const float4 *b, __global const float4 *c, float s)
{
    size_t gid = get_global_id(0);
    a[gid] = b[gid] + s * c[gid];
}
)cl";

// Defaults - override via CLI
constexpr size_t   DEFAULT_WORK_ITEMS  = size_t{64} * 1024 * 1024;
constexpr int      DEFAULT_ITERS       = 512; // about 50 ms per dispatch at 12 TFLOP/s
constexpr size_t   DEFAULT_LOCAL_SIZE  = 256;
constexpr size_t   DEFAULT_BUFFER_MIB  = 512;
constexpr auto     DEFAULT_PERIOD      = std::chrono::milliseconds{100};
constexpr cl_float STRESS_M            = 1.00001F;
constexpr cl_float STRESS_C            = 0.000003F;
constexpr cl_float NEVER_MATCHES       = -1.0F; // read_mem's sentinel; buffers hold values in [0, 1)
constexpr cl_uint  WRITE_SEED          = 0x5eedU;
constexpr cl_float TRIAD_SCALE         = 3.0F;
constexpr size_t   FLOAT4_BYTES        = 16;
constexpr double   FLOPS_PER_ITER      = 16.0; // 8 mads per work item per iteration

// Steady load: dispatches at most this long, batched to about BATCH_TARGET per queue.
constexpr auto STEADY_DISPATCH_TARGET = std::chrono::milliseconds{50};
constexpr auto BATCH_TARGET           = std::chrono::milliseconds{250};
constexpr int  MAX_AUTO_BATCH         = 4096;
// Duty cycle: dispatches at most period / DUTY_SLICES, so the busy share lands close to the target.
constexpr int  DUTY_SLICES            = 20;
constexpr int  MAX_SHRINKS            = 40;

constexpr auto WATCHDOG_POLL   = std::chrono::milliseconds{100};
constexpr auto STOP_GRACE      = std::chrono::seconds{3};
constexpr auto STATUS_INTERVAL = std::chrono::seconds{1};
constexpr auto MAIN_POLL       = std::chrono::milliseconds{100};

// RAII wrappers for OpenCL resources
struct OpenCLDeleter {
    void operator()(cl_context ctx) const
    {
        if (ctx != nullptr) { clReleaseContext(ctx); }
    }
    void operator()(cl_command_queue queue) const
    {
        if (queue != nullptr) { clReleaseCommandQueue(queue); }
    }
    void operator()(cl_program prog) const
    {
        if (prog != nullptr) { clReleaseProgram(prog); }
    }
    void operator()(cl_kernel kernel) const
    {
        if (kernel != nullptr) { clReleaseKernel(kernel); }
    }
    void operator()(cl_mem mem) const
    {
        if (mem != nullptr) { clReleaseMemObject(mem); }
    }
};

using ContextPtr = std::unique_ptr<std::remove_pointer_t<cl_context>,       OpenCLDeleter>;
using QueuePtr   = std::unique_ptr<std::remove_pointer_t<cl_command_queue>, OpenCLDeleter>;
using ProgramPtr = std::unique_ptr<std::remove_pointer_t<cl_program>,       OpenCLDeleter>;
using KernelPtr  = std::unique_ptr<std::remove_pointer_t<cl_kernel>,        OpenCLDeleter>;
using MemPtr     = std::unique_ptr<std::remove_pointer_t<cl_mem>,           OpenCLDeleter>;

// Exception for OpenCL errors
class OpenCLError : public std::runtime_error {
public:
    OpenCLError(std::string_view msg, cl_int code)
        : std::runtime_error(std::format("{} (error code: {})", msg, code))
        , errorCode(code) {}

    [[nodiscard]] cl_int code() const noexcept { return errorCode; }

private:
    cl_int errorCode;
};

void check(cl_int err, std::string_view what)
{
    if (err != CL_SUCCESS) { throw OpenCLError(what, err); }
}

// Modes
enum class Mode { Compute, Read, Write, Memory, Blit, H2D, D2H, Pcie, Mixed };

struct ModeInfo {
    Mode             mode;
    std::string_view name;
    std::string_view what;
    int              largeBuffers; // device-sized buffers it allocates, for sizing
};

constexpr std::array MODES{
    ModeInfo{Mode::Compute, "compute", "FP32 multiply-add loop in registers: ALU bound, no memory traffic", 0},
    ModeInfo{Mode::Read,    "read",    "streams a large buffer in: memory-read bound", 1},
    ModeInfo{Mode::Write,   "write",   "streams a large buffer out: memory-write bound", 1},
    ModeInfo{Mode::Memory,  "memory",  "triad a = b + s*c: reads two buffers, writes one", 3},
    ModeInfo{Mode::Blit,    "blit",    "device-to-device buffer copies; the driver picks a copy engine or a kernel", 2},
    ModeInfo{Mode::H2D,     "h2d",     "host-to-device copies from pinned memory: bus bound, one way", 2},
    ModeInfo{Mode::D2H,     "d2h",     "device-to-host copies into pinned memory: bus bound, the other way", 2},
    ModeInfo{Mode::Pcie,    "pcie",    "h2d and d2h at once: both bus directions", 4},
    ModeInfo{Mode::Mixed,   "mixed",   "compute on one queue and triad on the other", 3},
};

[[nodiscard]] const ModeInfo &modeInfo(Mode mode)
{
    for (const auto &m : MODES) {
        if (m.mode == mode) { return m; }
    }
    throw std::logic_error("unknown mode");
}

// GPU information structure
struct GpuEntry {
    cl_platform_id platform;
    cl_device_id   device;
    std::string    platformName;
    std::string    deviceName;
    cl_ulong       globalMem;
    cl_ulong       maxAlloc;
    cl_uint        computeUnits;
    size_t         maxWorkGroup;
    std::string    clVersion;     // CL_DEVICE_VERSION, e.g. "OpenCL 3.0 NEO"
    std::string    driverVersion; // CL_DRIVER_VERSION
};

// Command line arguments
struct Arguments {
    bool                                listOnly{false};
    bool                                allDevices{false};
    std::vector<int>                    devices;  // empty: the default device
    Mode                                mode{Mode::Compute};
    int                                 duty{100};
    std::optional<std::chrono::seconds> ramp;
    std::chrono::milliseconds           period{DEFAULT_PERIOD};
    std::optional<std::chrono::seconds> duration;
    size_t                              bufferMiB{DEFAULT_BUFFER_MIB};
    int                                 iters{DEFAULT_ITERS};
    size_t                              workItems{DEFAULT_WORK_ITEMS};
    size_t                              localSize{DEFAULT_LOCAL_SIZE}; // Xe2 requires a non-null local size
    std::optional<int>                  batch;    // nullopt: sized from the calibrated dispatch
    bool                                calibrate{true};
    int                                 timeoutSecs{0}; // 0 = none; >0 = abort after N seconds for a core dump
    bool                                quiet{false};
    bool                                verbose{false}; // print each device's setup stages

    [[nodiscard]] bool periodic() const noexcept { return duty < 100 || ramp.has_value(); }
};

void printHelp()
{
    std::cout <<
        "gpu_stress - generate a chosen kind of GPU load\n\n"
        "  --list               List OpenCL GPUs with their indices and exit\n"
        "  --device N[,M...]    GPUs to load, by --list index (default: first Intel GPU, else 0)\n"
        "  --device all         Every GPU, one thread each\n"
        "  --mode MODE          What to load (default: compute); see below\n"
        "  --duty PCT           Keep the GPU busy PCT% of each period (1-100, default 100)\n"
        "  --ramp SEC           Sweep the duty 0 -> 100 -> 0% over SEC seconds, repeating\n"
        "  --period-ms N        Duty-cycle period (10-10000, default 100)\n"
        "  --duration SEC       Stop after SEC seconds (default: run until Ctrl-C)\n"
        "  --buffer-mib N       Size of each large buffer (default 512, capped by the device)\n"
        "  --iters N            compute: multiply-add iterations per work item (default 512)\n"
        "  --work-items N       compute: global work items, a multiple of --local-size (default 64M)\n"
        "  --local-size N       Work-group size (default 256, capped by the device)\n"
        "  --batch N            Dispatches per queue between syncs (default: about 250 ms of work)\n"
        "  --no-calibrate       Keep the sizes given instead of shrinking long dispatches\n"
        "  --timeout SEC        abort() after SEC seconds, for a core dump of a hang\n"
        "  --quiet              No live status line, just the summary\n"
        "  --verbose            Print each device's setup stages as they finish, to find one that stalls\n\n"
        "Modes:\n";
    for (const auto &m : MODES) {
        std::cout << std::format("  {:<8} {}\n", m.name, m.what);
    }
    std::cout <<
        "\nExamples:\n"
        "  Compute (ALU) bound, full load until Ctrl-C:\n"
        "    gpu_stress --mode compute\n"
        "  Memory-read bound for 60 s:\n"
        "    gpu_stress --mode read --duration 60\n"
        "  Memory-write bound:\n"
        "    gpu_stress --mode write\n"
        "  Memory-bandwidth bound, reads and writes together (2:1):\n"
        "    gpu_stress --mode memory\n"
        "  Copy bound, device-to-device:\n"
        "    gpu_stress --mode blit\n"
        "  Bus bound, host to device / device to host / both ways:\n"
        "    gpu_stress --mode h2d\n"
        "    gpu_stress --mode d2h\n"
        "    gpu_stress --mode pcie\n"
        "  Compute and memory traffic at once:\n"
        "    gpu_stress --mode mixed\n"
        "  Partial load, busy 30% of the time:\n"
        "    gpu_stress --mode compute --duty 30\n"
        "  Coarser on/off pattern, 2 s on, 2 s off:\n"
        "    gpu_stress --mode compute --duty 50 --period-ms 4000\n"
        "  Sweep load 0 -> 100 -> 0% every 20 s, for watching graphs:\n"
        "    gpu_stress --mode read --ramp 20\n"
        "  Every GPU at once, or two chosen by --list index:\n"
        "    gpu_stress --device all --mode compute\n"
        "    gpu_stress --device 0,1 --mode memory\n"
        "  Large single dispatches, exactly as sized (the old default; each can take seconds,\n"
        "  and Ctrl-C then waits up to 3 s and exits without a summary):\n"
        "    gpu_stress --mode compute --iters 65536 --no-calibrate\n"
        "  Thermal soak, summary only, abort for a core dump if it hangs past 10 min:\n"
        "    gpu_stress --mode compute --duration 600 --quiet --timeout 660\n";
}

// Helper to get platform/device info as string
template <typename InfoFunc, typename Handle>
std::string getInfoString(InfoFunc func, Handle handle, cl_uint param)
{
    size_t size = 0;
    func(handle, param, 0, nullptr, &size);
    if (size == 0) { return {}; }

    std::string result(size, '\0');
    func(handle, param, size, result.data(), nullptr);
    if (!result.empty() && result.back() == '\0') {
        result.pop_back();
    }
    return result;
}

template <typename T>
T deviceInfo(cl_device_id device, cl_device_info param)
{
    T value{};
    clGetDeviceInfo(device, param, sizeof(T), &value, nullptr);
    return value;
}

// One installed OpenCL platform, for --list: shows a driver that is present but offers no GPU.
struct PlatformReport {
    std::string           name;
    std::string           version;
    cl_uint               gpus{0};
    cl_uint               otherDevices{0}; // CPU and accelerator devices, not loaded
    std::optional<cl_int> gpuQueryError;   // clGetDeviceIDs(GPU) failed with this code
};

// Every platform the OpenCL ICD loader found; empty with @p loaderError set when it found none.
std::vector<PlatformReport> reportPlatforms(cl_int &loaderError)
{
    std::vector<PlatformReport> out;
    cl_uint count = 0;
    loaderError = clGetPlatformIDs(0, nullptr, &count);
    if (loaderError != CL_SUCCESS || count == 0) {
        return out;
    }
    std::vector<cl_platform_id> platforms(count);
    clGetPlatformIDs(count, platforms.data(), nullptr);
    for (auto *const platform : platforms) {
        PlatformReport r{
            .name          = getInfoString(clGetPlatformInfo, platform, CL_PLATFORM_NAME),
            .version       = getInfoString(clGetPlatformInfo, platform, CL_PLATFORM_VERSION),
            .gpus          = 0,
            .otherDevices  = 0,
            .gpuQueryError = std::nullopt,
        };
        if (const cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &r.gpus);
            err != CL_SUCCESS) {
            r.gpus = 0;
            if (err != CL_DEVICE_NOT_FOUND) { r.gpuQueryError = err; }
        }
        cl_uint all = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &all) == CL_SUCCESS && all > r.gpus) {
            r.otherDevices = all - r.gpus;
        }
        out.push_back(std::move(r));
    }
    return out;
}

// Enumerate all available OpenCL GPUs
std::vector<GpuEntry> enumerateGpus()
{
    std::vector<GpuEntry> gpus;

    cl_uint platformCount = 0;
    if (clGetPlatformIDs(0, nullptr, &platformCount) != CL_SUCCESS || platformCount == 0) {
        return gpus;
    }

    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    for (auto *const platform : platforms) {
        cl_uint deviceCount = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount) != CL_SUCCESS) {
            continue;
        }

        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr);

        const auto platformName = getInfoString(clGetPlatformInfo, platform, CL_PLATFORM_NAME);

        for (auto *const device : devices) {
            gpus.push_back({
                .platform     = platform,
                .device       = device,
                .platformName = platformName,
                .deviceName   = getInfoString(clGetDeviceInfo, device, CL_DEVICE_NAME),
                .globalMem    = deviceInfo<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE),
                .maxAlloc     = deviceInfo<cl_ulong>(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE),
                .computeUnits = deviceInfo<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS),
                .maxWorkGroup = deviceInfo<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE),
                .clVersion     = getInfoString(clGetDeviceInfo, device, CL_DEVICE_VERSION),
                .driverVersion = getInfoString(clGetDeviceInfo, device, CL_DRIVER_VERSION),
            });
        }
    }

    return gpus;
}

// Parse command line arguments
Arguments parseArguments(std::span<char *> args)
{
    Arguments result;

    auto requireNext = [&](size_t i, std::string_view flag) -> std::string_view {
        if (i + 1 >= args.size()) {
            throw std::invalid_argument(std::format("{} requires a value", flag));
        }
        return args[i + 1];
    };

    // Consume the next token as an int in [lo, hi].
    auto nextInt = [&](size_t &i, std::string_view flag, int lo, int hi) -> int {
        const int n = std::stoi(std::string{requireNext(i++, flag)});
        if (n < lo || n > hi) { throw std::invalid_argument(std::format("{} must be {}-{}", flag, lo, hi)); }
        return n;
    };
    auto nextPosInt = [&](size_t &i, std::string_view flag) -> int {
        return nextInt(i, flag, 1, std::numeric_limits<int>::max());
    };
    auto nextPosULL = [&](size_t &i, std::string_view flag) -> size_t {
        const auto n = std::stoull(std::string{requireNext(i++, flag)});
        if (n == 0) { throw std::invalid_argument(std::format("{} must be positive", flag)); }
        return static_cast<size_t>(n);
    };

    for (size_t i = 1; i < args.size(); ++i) {
        const std::string_view arg = args[i];

        if (arg == "--help" || arg == "-h") {
            printHelp();
            std::exit(0);
        } else if (arg == "--list") {
            result.listOnly = true;
        } else if (arg == "--device") {
            const std::string_view spec = requireNext(i++, arg);
            if (spec == "all") {
                result.allDevices = true;
                continue;
            }
            for (size_t start = 0; start <= spec.size();) {
                const size_t comma = std::min(spec.find(',', start), spec.size());
                result.devices.push_back(std::stoi(std::string{spec.substr(start, comma - start)}));
                start = comma + 1;
            }
        } else if (arg == "--mode") {
            const std::string_view name = requireNext(i++, arg);
            const auto it = std::ranges::find(MODES, name, &ModeInfo::name);
            if (it == MODES.end()) {
                throw std::invalid_argument(std::format("Unknown mode '{}'; see --help", name));
            }
            result.mode = it->mode;
        } else if (arg == "--duty") {
            result.duty = nextInt(i, arg, 1, 100);
        } else if (arg == "--ramp") {
            result.ramp = std::chrono::seconds{nextPosInt(i, arg)};
        } else if (arg == "--period-ms") {
            result.period = std::chrono::milliseconds{nextInt(i, arg, 10, 10000)};
        } else if (arg == "--duration") {
            result.duration = std::chrono::seconds{nextPosInt(i, arg)};
        } else if (arg == "--buffer-mib") {
            result.bufferMiB = nextPosULL(i, arg);
        } else if (arg == "--iters") {
            result.iters = nextPosInt(i, arg);
        } else if (arg == "--batch") {
            result.batch = nextPosInt(i, arg);
        } else if (arg == "--work-items") {
            result.workItems = nextPosULL(i, arg);
        } else if (arg == "--local-size") {
            result.localSize = nextPosULL(i, arg);
        } else if (arg == "--no-calibrate") {
            result.calibrate = false;
        } else if (arg == "--timeout") {
            result.timeoutSecs = nextPosInt(i, arg);
        } else if (arg == "--quiet") {
            result.quiet = true;
        } else if (arg == "--verbose") {
            result.verbose = true;
        } else {
            throw std::invalid_argument(std::format("Unknown argument: {} (see --help)", arg));
        }
    }

    if (result.workItems % result.localSize != 0) {
        throw std::invalid_argument(std::format("--work-items ({}) must be a multiple of --local-size ({})",
                                                result.workItems, result.localSize));
    }
    return result;
}

// List all available GPUs, then every OpenCL platform, so a missing driver is visible too
void listGpus(const std::vector<GpuEntry> &gpus)
{
    std::cout << std::format("{:<4}  {:<36}  {:>9}  {:>9}  {:>4}  {}\n", "Idx", "Device", "Memory", "MaxAlloc",
                             "CUs", "Platform");
    for (size_t idx = 0; const auto &gpu : gpus) {
        std::cout << std::format("{:<4}  {:<36}  {:>5} MiB  {:>5} MiB  {:>4}  {}\n", idx, gpu.deviceName,
                                 gpu.globalMem >> 20, gpu.maxAlloc >> 20, gpu.computeUnits, gpu.platformName);
        std::cout << std::format("      {}, driver {}\n", gpu.clVersion, gpu.driverVersion);
        ++idx;
    }

    cl_int loaderError = CL_SUCCESS;
    const auto platforms = reportPlatforms(loaderError);
    std::cout << "\nOpenCL platforms (one per installed driver; a GPU missing above needs its driver here):\n";
    if (platforms.empty()) {
        std::cout << std::format("  none found (clGetPlatformIDs returned {}): no OpenCL driver (ICD) is installed\n",
                                 loaderError);
        return;
    }
    for (const auto &p : platforms) {
        std::string devices = std::format("{} GPU", p.gpus);
        if (p.otherDevices > 0) { devices += std::format(", {} other", p.otherDevices); }
        if (p.gpuQueryError) { devices += std::format(" (GPU query failed: {})", *p.gpuQueryError); }
        std::cout << std::format("  {:<40}  {:<24}  {}\n", p.name, p.version, devices);
    }
}

// The --list indices to run on.
std::vector<int> selectDevices(const std::vector<GpuEntry> &gpus, const Arguments &args)
{
    std::vector<int> out;
    if (args.allDevices) {
        for (int i = 0; i < static_cast<int>(gpus.size()); ++i) { out.push_back(i); }
        return out;
    }
    for (const int idx : args.devices) {
        if (idx < 0 || static_cast<size_t>(idx) >= gpus.size()) {
            throw std::out_of_range(std::format("--device {} out of range (0-{}); use --list to see available GPUs",
                                                idx, gpus.size() - 1));
        }
        if (std::ranges::find(out, idx) == out.end()) { out.push_back(idx); }
    }
    if (!out.empty()) { return out; }

    // Default: first Intel GPU, otherwise device 0
    for (size_t idx = 0; const auto &gpu : gpus) {
        if (gpu.platformName.find("Intel") != std::string::npos ||
            gpu.deviceName.find("Intel") != std::string::npos) {
            return {static_cast<int>(idx)};
        }
        ++idx;
    }
    return {0};
}

// Build OpenCL program
ProgramPtr buildProgram(cl_context ctx, cl_device_id device)
{
    cl_int err = CL_SUCCESS;
    const char  *src    = KERNEL_SRC.data();
    const size_t srcLen = KERNEL_SRC.size();

    ProgramPtr program(clCreateProgramWithSource(ctx, 1, &src, &srcLen, &err));
    check(err, "Failed to create program");

    err = clBuildProgram(program.get(), 1, &device, "-cl-mad-enable", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program.get(), device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string buildLog(logSize, '\0');
        clGetProgramBuildInfo(program.get(), device, CL_PROGRAM_BUILD_LOG,
                              logSize, buildLog.data(), nullptr);
        const std::lock_guard lock{g_print};
        std::cerr << "Build log:\n" << buildLog << '\n';
        throw OpenCLError("Failed to build program", err);
    }

    return program;
}

KernelPtr createKernel(cl_program program, const char *name)
{
    cl_int err = CL_SUCCESS;
    KernelPtr kernel(clCreateKernel(program, name, &err));
    check(err, std::format("Failed to create kernel {}", name));
    return kernel;
}

template <typename T>
void setArg(cl_kernel kernel, cl_uint index, const T &value)
{
    check(clSetKernelArg(kernel, index, sizeof(T), &value), std::format("Failed to set kernel arg {}", index));
}

MemPtr createBuffer(cl_context ctx, cl_mem_flags flags, size_t bytes)
{
    cl_int err = CL_SUCCESS;
    MemPtr mem(clCreateBuffer(ctx, flags, bytes, nullptr, &err));
    check(err, std::format("Failed to create a {} MiB buffer", bytes >> 20));
    return mem;
}

void enqueueRange(cl_command_queue queue, cl_kernel kernel, size_t global, size_t local)
{
    const std::array<size_t, 1> g = {global};
    const std::array<size_t, 1> l = {local};
    check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, g.data(), l.data(), 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel failed");
}

// What one dispatch moves or computes, for the throughput figures.
struct Work {
    double bytes{0.0};
    double flops{0.0};
};

// One kind of dispatch. Every op can halve its own size, which is how calibration works.
class Op {
public:
    Op() = default;
    virtual ~Op() = default;
    Op(const Op &) = delete;
    Op &operator=(const Op &) = delete;
    Op(Op &&) = delete;
    Op &operator=(Op &&) = delete;

    virtual void enqueue(cl_command_queue queue) = 0;
    /// Halves the work of one dispatch; false when it is as small as it can get.
    virtual bool shrink() = 0;
    [[nodiscard]] virtual Work perDispatch() const = 0;
    [[nodiscard]] virtual std::string describe() const = 0;
};

// Halves a work size that must stay a positive multiple of @p step.
bool halveMultiple(size_t &n, size_t step)
{
    const size_t half = n / 2 / step * step;
    if (half == 0 || half == n) { return false; }
    n = half;
    return true;
}

class ComputeOp final : public Op {
public:
    ComputeOp(cl_context ctx, cl_program program, size_t items, size_t local, int iterations)
        : kernel(createKernel(program, "stress"))
        , out(createBuffer(ctx, CL_MEM_WRITE_ONLY, items * sizeof(cl_float)))
        , workItems(items), localSize(local), iters(iterations)
    {
        cl_mem buf = out.get();
        setArg(kernel.get(), 0, buf);
        setArg(kernel.get(), 1, static_cast<cl_int>(iters));
        setArg(kernel.get(), 2, STRESS_M);
        setArg(kernel.get(), 3, STRESS_C);
    }

    void enqueue(cl_command_queue queue) override { enqueueRange(queue, kernel.get(), workItems, localSize); }

    bool shrink() override
    {
        if (iters > 1) {
            iters /= 2;
            setArg(kernel.get(), 1, static_cast<cl_int>(iters));
            return true;
        }
        return halveMultiple(workItems, localSize);
    }

    [[nodiscard]] Work perDispatch() const override
    {
        return {.bytes = 0.0, .flops = static_cast<double>(workItems) * iters * FLOPS_PER_ITER};
    }

    [[nodiscard]] std::string describe() const override
    {
        return std::format("compute {} items x {} iters", workItems, iters);
    }

private:
    KernelPtr kernel;
    MemPtr    out;
    size_t    workItems;
    size_t    localSize;
    int       iters;
};

enum class Stream { Read, Write, Triad };

class StreamOp final : public Op {
public:
    // @p buffers: Read {in, sink}; Write {out}; Triad {a, b, c}. Not owned.
    StreamOp(cl_program program, Stream kind, std::span<const cl_mem> buffers, size_t elements, size_t local)
        : kernel(createKernel(program, kind == Stream::Read ? "read_mem" : kind == Stream::Write ? "write_mem" : "triad"))
        , stream(kind), count(elements), localSize(local)
    {
        for (cl_uint i = 0; i < buffers.size(); ++i) { setArg(kernel.get(), i, buffers[i]); }
        const auto last = static_cast<cl_uint>(buffers.size());
        if (kind == Stream::Write) {
            setArg(kernel.get(), last, WRITE_SEED);
        } else {
            setArg(kernel.get(), last, kind == Stream::Read ? NEVER_MATCHES : TRIAD_SCALE);
        }
    }

    void enqueue(cl_command_queue queue) override { enqueueRange(queue, kernel.get(), count, localSize); }
    bool shrink() override { return halveMultiple(count, localSize); }

    [[nodiscard]] Work perDispatch() const override
    {
        const double bytes = static_cast<double>(count * FLOAT4_BYTES);
        return {.bytes = stream == Stream::Triad ? 3.0 * bytes : bytes, .flops = 0.0};
    }

    [[nodiscard]] std::string describe() const override
    {
        const std::string_view name = stream == Stream::Read ? "read" : stream == Stream::Write ? "write" : "triad";
        return std::format("{} {} MiB", name, count * FLOAT4_BYTES >> 20);
    }

private:
    KernelPtr kernel;
    Stream    stream;
    size_t    count; // float4 elements per dispatch
    size_t    localSize;
};

class BlitOp final : public Op {
public:
    BlitOp(cl_mem source, cl_mem destination, size_t size) : src(source), dst(destination), bytes(size) {}

    void enqueue(cl_command_queue queue) override
    {
        check(clEnqueueCopyBuffer(queue, src, dst, 0, 0, bytes, 0, nullptr, nullptr), "clEnqueueCopyBuffer failed");
    }
    bool shrink() override { return halveMultiple(bytes, FLOAT4_BYTES); }
    // Counted as memory traffic: every byte is read once and written once.
    [[nodiscard]] Work perDispatch() const override { return {.bytes = 2.0 * static_cast<double>(bytes), .flops = 0.0}; }
    [[nodiscard]] std::string describe() const override { return std::format("blit {} MiB", bytes >> 20); }

private:
    cl_mem src;
    cl_mem dst;
    size_t bytes;
};

// Pinned host memory, mapped for as long as it lives.
class PinnedHost {
public:
    PinnedHost(cl_context ctx, cl_command_queue mapQueue, size_t size)
        : buffer(createBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size)), queue(mapQueue)
    {
        cl_int err = CL_SUCCESS;
        ptr = clEnqueueMapBuffer(queue, buffer.get(), CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, nullptr,
                                 nullptr, &err);
        check(err, "Failed to map pinned host memory");
        std::fill_n(static_cast<unsigned char *>(ptr), size, static_cast<unsigned char>(0));
    }
    ~PinnedHost()
    {
        if (ptr != nullptr) {
            clEnqueueUnmapMemObject(queue, buffer.get(), ptr, 0, nullptr, nullptr);
            clFinish(queue);
        }
    }
    PinnedHost(const PinnedHost &) = delete;
    PinnedHost &operator=(const PinnedHost &) = delete;
    PinnedHost(PinnedHost &&) = delete;
    PinnedHost &operator=(PinnedHost &&) = delete;

    [[nodiscard]] void *data() const noexcept { return ptr; }

private:
    MemPtr           buffer;
    cl_command_queue queue; // not owned; must outlive this
    void            *ptr{nullptr};
};

class TransferOp final : public Op {
public:
    TransferOp(bool toDevice, cl_mem device, void *host, size_t size)
        : hostToDevice(toDevice), dev(device), hostPtr(host), bytes(size) {}

    void enqueue(cl_command_queue queue) override
    {
        const cl_int err = hostToDevice
            ? clEnqueueWriteBuffer(queue, dev, CL_FALSE, 0, bytes, hostPtr, 0, nullptr, nullptr)
            : clEnqueueReadBuffer(queue, dev, CL_FALSE, 0, bytes, hostPtr, 0, nullptr, nullptr);
        check(err, hostToDevice ? "clEnqueueWriteBuffer failed" : "clEnqueueReadBuffer failed");
    }
    bool shrink() override { return halveMultiple(bytes, FLOAT4_BYTES); }
    [[nodiscard]] Work perDispatch() const override { return {.bytes = static_cast<double>(bytes), .flops = 0.0}; }
    [[nodiscard]] std::string describe() const override
    {
        return std::format("{} {} MiB", hostToDevice ? "h2d" : "d2h", bytes >> 20);
    }

private:
    bool   hostToDevice;
    cl_mem dev;
    void  *hostPtr;
    size_t bytes;
};

// Live counters for one device, shared between its worker thread and the status line.
struct DeviceStats {
    std::atomic<uint64_t> dispatches{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> flops{0};
    std::atomic<int>      dutyNow{100};
    std::atomic<bool>     ready{false};
    std::atomic<bool>     done{false};
    std::string           error; // written once, before done
};

// One command queue and what it dispatches, with what is in flight since the last sync.
struct Lane {
    cl_command_queue queue;
    Op              *op;
    uint64_t         pending{0};
    Work             pendingWork{};
};

void submit(Lane &lane, int count)
{
    const Work w = lane.op->perDispatch();
    for (int i = 0; i < count; ++i) { lane.op->enqueue(lane.queue); }
    lane.pending += static_cast<uint64_t>(count);
    lane.pendingWork.bytes += w.bytes * count;
    lane.pendingWork.flops += w.flops * count;
}

void complete(Lane &lane, DeviceStats &stats)
{
    check(clFinish(lane.queue), "clFinish failed");
    stats.dispatches += lane.pending;
    stats.bytes += static_cast<uint64_t>(lane.pendingWork.bytes);
    stats.flops += static_cast<uint64_t>(lane.pendingWork.flops);
    lane.pending = 0;
    lane.pendingWork = {};
}

Seconds timeOne(Lane &lane)
{
    const auto start = Clock::now();
    lane.op->enqueue(lane.queue);
    check(clFinish(lane.queue), "clFinish failed");
    return Clock::now() - start;
}

// Shrinks the lane's op until one dispatch takes at most @p target. Returns the final time.
// Each pass halves as often as the measured time says it needs, so even a dispatch of seconds
// takes only a few passes.
Seconds calibrate(Lane &lane, Seconds target, bool allowed)
{
    (void)timeOne(lane); // the first dispatch pays for compilation and page faults
    Seconds t = timeOne(lane);
    for (int pass = 0; allowed && pass < MAX_SHRINKS && t > target && !g_stop.load(std::memory_order_acquire);
         ++pass) {
        const auto halvings = static_cast<int>(std::ceil(std::log2(t / target)));
        bool shrunk = false;
        for (int i = 0; i < std::max(1, halvings) && lane.op->shrink(); ++i) { shrunk = true; }
        if (!shrunk) { break; }
        t = timeOne(lane);
    }
    return t;
}

int dutyAt(const Arguments &args, Clock::duration elapsed)
{
    if (!args.ramp) { return args.duty; }
    const double span = Seconds{*args.ramp}.count();
    const double phase = std::fmod(Seconds{elapsed}.count(), span) / span;
    const double triangle = phase < 0.5 ? 2.0 * phase : 2.0 * (1.0 - phase);
    return static_cast<int>(std::lround(triangle * 100.0));
}

void runSteady(std::array<Lane, 2> &lanes, int batch, DeviceStats &stats)
{
    submit(lanes[0], batch);
    submit(lanes[1], batch);
    while (!g_stop.load(std::memory_order_acquire)) {
        for (auto &lane : lanes) {
            complete(lane, stats);
            if (!g_stop.load(std::memory_order_acquire)) { submit(lane, batch); }
        }
    }
}

void runPeriodic(std::array<Lane, 2> &lanes, const Arguments &args, DeviceStats &stats)
{
    const auto start = Clock::now();
    while (!g_stop.load(std::memory_order_acquire)) {
        const auto periodStart = Clock::now();
        const int duty = dutyAt(args, periodStart - start);
        stats.dutyNow = duty;
        const auto busyEnd = periodStart + args.period * duty / 100;
        if (duty > 0) {
            // Two dispatches in flight, one per queue, until the busy share of the period is used.
            submit(lanes[0], 1);
            submit(lanes[1], 1);
            while (Clock::now() < busyEnd && !g_stop.load(std::memory_order_acquire)) {
                for (auto &lane : lanes) {
                    complete(lane, stats);
                    if (Clock::now() < busyEnd) { submit(lane, 1); }
                }
            }
            complete(lanes[0], stats);
            complete(lanes[1], stats);
        }
        const auto periodEnd = periodStart + args.period;
        while (Clock::now() < periodEnd && !g_stop.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::min<Clock::duration>(periodEnd - Clock::now(), WATCHDOG_POLL));
        }
    }
}

// Sets up and runs the load on one device until g_stop. Errors end up in stats.error.
void runDevice(int index, const GpuEntry &gpu, const Arguments &args, DeviceStats &stats)
{
    const auto started = Clock::now();
    const auto say = [&](const std::string &text) {
        const std::lock_guard lock{g_print};
        std::cout << std::format("[{}] {}\n", index, text) << std::flush;
    };
    // --verbose: each setup stage as it finishes, so a device that stalls shows where.
    const auto stage = [&](std::string_view what) {
        if (args.verbose) { say(std::format("{:7.3f} s  {}", Seconds{Clock::now() - started}.count(), what)); }
    };
    say(std::format("{} - {}, {}, driver {}", gpu.deviceName, gpu.platformName, gpu.clVersion, gpu.driverVersion));
    try {
        cl_int err = CL_SUCCESS;
        ContextPtr context(clCreateContext(nullptr, 1, &gpu.device, nullptr, nullptr, &err));
        check(err, "Failed to create context");
        stage("context created");
        std::array<QueuePtr, 2> queues;
        for (auto &q : queues) {
            q.reset(clCreateCommandQueueWithProperties(context.get(), gpu.device, nullptr, &err));
            check(err, "Failed to create command queue");
        }
        stage("command queues created");
        const auto program = buildProgram(context.get(), gpu.device);
        stage("kernels built");

        const size_t local = std::min(args.localSize, gpu.maxWorkGroup);
        if (args.workItems % local != 0) {
            throw std::invalid_argument(std::format("--work-items must be a multiple of the device's local size {}", local));
        }
        // Large buffers: the requested size, within one allocation, and leaving most of memory free.
        const ModeInfo &mode = modeInfo(args.mode);
        const size_t buffersWanted = static_cast<size_t>(std::max(1, mode.largeBuffers));
        const size_t step = FLOAT4_BYTES * local;
        size_t bufferBytes = std::min<size_t>({args.bufferMiB << 20, static_cast<size_t>(gpu.maxAlloc),
                                               static_cast<size_t>(gpu.globalMem) / (2 * buffersWanted)});
        bufferBytes = bufferBytes / step * step;
        if (mode.largeBuffers > 0 && bufferBytes == 0) {
            throw std::runtime_error("device memory too small for the buffers this mode needs");
        }
        const size_t elements = bufferBytes / FLOAT4_BYTES;

        std::vector<MemPtr> buffers;
        const auto fill = createKernel(program.get(), "fill_noise");
        auto deviceBuffer = [&]() -> cl_mem {
            buffers.push_back(createBuffer(context.get(), CL_MEM_READ_WRITE, bufferBytes));
            cl_mem buf = buffers.back().get();
            setArg(fill.get(), 0, buf);
            setArg(fill.get(), 1, static_cast<cl_uint>(buffers.size()));
            enqueueRange(queues[0].get(), fill.get(), elements, local);
            return buf;
        };
        std::vector<std::unique_ptr<PinnedHost>> pinned;
        auto hostBuffer = [&]() -> void * {
            pinned.push_back(std::make_unique<PinnedHost>(context.get(), queues[0].get(), bufferBytes));
            return pinned.back()->data();
        };
        MemPtr sink = createBuffer(context.get(), CL_MEM_WRITE_ONLY, sizeof(cl_float) * 4);

        auto triadOp = [&] {
            const std::array<cl_mem, 3> abc{deviceBuffer(), deviceBuffer(), deviceBuffer()};
            return std::make_unique<StreamOp>(program.get(), Stream::Triad, abc, elements, local);
        };
        std::array<std::unique_ptr<Op>, 2> ops;
        switch (args.mode) {
        case Mode::Compute:
            ops[0] = std::make_unique<ComputeOp>(context.get(), program.get(), args.workItems, local, args.iters);
            break;
        case Mode::Read: {
            const std::array<cl_mem, 2> in{deviceBuffer(), sink.get()};
            ops[0] = std::make_unique<StreamOp>(program.get(), Stream::Read, in, elements, local);
            break;
        }
        case Mode::Write: {
            const std::array<cl_mem, 1> out{deviceBuffer()};
            ops[0] = std::make_unique<StreamOp>(program.get(), Stream::Write, out, elements, local);
            break;
        }
        case Mode::Memory:
            ops[0] = triadOp();
            break;
        case Mode::Blit: {
            cl_mem src = deviceBuffer();
            ops[0] = std::make_unique<BlitOp>(src, deviceBuffer(), bufferBytes);
            break;
        }
        case Mode::H2D:
        case Mode::D2H: {
            const bool toDevice = args.mode == Mode::H2D;
            ops[0] = std::make_unique<TransferOp>(toDevice, deviceBuffer(), hostBuffer(), bufferBytes);
            // Two queues alternating on one transfer would only take turns; give each its own buffers.
            ops[1] = std::make_unique<TransferOp>(toDevice, deviceBuffer(), hostBuffer(), bufferBytes);
            break;
        }
        case Mode::Pcie:
            ops[0] = std::make_unique<TransferOp>(true, deviceBuffer(), hostBuffer(), bufferBytes);
            ops[1] = std::make_unique<TransferOp>(false, deviceBuffer(), hostBuffer(), bufferBytes);
            break;
        case Mode::Mixed:
            ops[0] = std::make_unique<ComputeOp>(context.get(), program.get(), args.workItems, local, args.iters);
            ops[1] = triadOp();
            break;
        }
        check(clFinish(queues[0].get()), "Failed to initialise buffers");
        stage(std::format("buffers ready ({} MiB each)", bufferBytes >> 20));

        std::array<Lane, 2> lanes{
            Lane{.queue = queues[0].get(), .op = ops[0].get()},
            Lane{.queue = queues[1].get(), .op = ops[1] ? ops[1].get() : ops[0].get()},
        };

        const Seconds target = args.periodic() ? Seconds{args.period} / DUTY_SLICES : Seconds{STEADY_DISPATCH_TARGET};
        Seconds dispatch = calibrate(lanes[0], target, args.calibrate);
        if (ops[1]) { dispatch = std::max(dispatch, calibrate(lanes[1], target, args.calibrate)); }
        const int batch = args.batch.value_or(std::clamp(
            static_cast<int>(Seconds{BATCH_TARGET} / std::max(dispatch, Seconds{1e-6})), 1, MAX_AUTO_BATCH));

        {
            const std::lock_guard lock{g_print};
            std::cout << std::format("[{}] {}{}  dispatch {:.1f} ms", index, ops[0]->describe(),
                                     ops[1] ? " + " + ops[1]->describe() : std::string{}, dispatch.count() * 1e3);
            if (args.periodic()) {
                std::cout << std::format("  period {} ms  {}\n", args.period.count(),
                                         args.ramp ? std::format("ramp {} s", args.ramp->count())
                                                   : std::format("duty {}%", args.duty));
            } else {
                std::cout << std::format("  batch {}\n", batch);
            }
            if (dispatch > target * 2) {
                std::cout << std::format("[{}] note: dispatches are {:.0f} ms, longer than the {:.0f} ms aimed for\n", index,
                                         dispatch.count() * 1e3, target.count() * 1e3);
            }
        }
        stats.ready = true;

        if (args.periodic()) {
            runPeriodic(lanes, args, stats);
        } else {
            runSteady(lanes, batch, stats);
        }
        // Drain; the watchdog ends the process if this takes too long.
        (void)clFinish(queues[0].get());
        (void)clFinish(queues[1].get());
    } catch (const std::exception &e) {
        stats.error = e.what();
        // Now, not only in the summary: with several devices the others keep running, and a
        // failed device would otherwise just read 0 on the status line.
        const std::lock_guard lock{g_print};
        std::cerr << std::format("\n[{}] Error: {}\n", index, stats.error) << std::flush;
    }
    stats.done = true;
}

std::string formatRates(double bytesPerSecond, double flopsPerSecond)
{
    std::string out;
    if (flopsPerSecond > 0.0) { out += std::format("{:7.2f} TFLOP/s", flopsPerSecond / 1e12); }
    if (bytesPerSecond > 0.0 || flopsPerSecond <= 0.0) {
        out += std::format("{}{:7.1f} GB/s", out.empty() ? "" : "  ", bytesPerSecond / 1e9);
    }
    return out;
}

} // anonymous namespace

int main(int argc, char **argv)
{
    try {
        (void)std::signal(SIGINT,  onSignal);
        (void)std::signal(SIGTERM, onSignal);

        const auto args = parseArguments(std::span(argv, static_cast<size_t>(argc)));

        const auto gpus = enumerateGpus();
        if (gpus.empty()) {
            std::cerr << "No OpenCL GPUs found.\n";
            return 1;
        }

        if (args.listOnly) {
            listGpus(gpus);
            return 0;
        }

        const auto selected = selectDevices(gpus, args);
        std::cout << std::format("Mode     : {} - {}\n", modeInfo(args.mode).name, modeInfo(args.mode).what);

        // Some drivers' clFinish blocks until the GPU completes and ignores SIGINT. After a stop,
        // give the workers STOP_GRACE to drain and print, then end the process regardless.
        std::thread([] {
            while (!g_stop.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(WATCHDOG_POLL);
            }
            std::this_thread::sleep_for(STOP_GRACE);
            _exit(0);
        }).detach();

        // Abort after N seconds to force a core dump for GPU hang diagnostics.
        if (args.timeoutSecs > 0) {
            std::thread([timeoutSecs = args.timeoutSecs]() {
                std::this_thread::sleep_for(std::chrono::seconds(timeoutSecs));
                std::cerr << "Timeout reached (" << timeoutSecs << "s), forcing abort for core dump...\n";
                std::cerr.flush();
                std::abort();
            }).detach();
        }

        std::vector<std::unique_ptr<DeviceStats>> stats;
        std::vector<std::thread> workers;
        for (const int idx : selected) {
            stats.push_back(std::make_unique<DeviceStats>());
            workers.emplace_back(runDevice, idx, std::cref(gpus[static_cast<size_t>(idx)]), std::cref(args),
                                 std::ref(*stats.back()));
        }

        const auto allDone = [&] { return std::ranges::all_of(stats, [](const auto &s) { return s->done.load(); }); };
        const auto anyReady = [&] { return std::ranges::any_of(stats, [](const auto &s) { return s->ready.load(); }); };
        while (!anyReady() && !allDone() && !g_stop.load()) {
            std::this_thread::sleep_for(MAIN_POLL);
        }

        // Live status: rates since the last line, per device.
        const auto start = Clock::now();
        auto lastLine = start;
        std::vector<std::pair<uint64_t, uint64_t>> last(stats.size(), {0, 0});
        if (!args.quiet && !allDone()) { std::cout << "Running - Ctrl-C to stop\n"; }
        while (!allDone() && !g_stop.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(MAIN_POLL);
            const auto now = Clock::now();
            if (args.duration && now - start >= *args.duration) {
                g_stop.store(true, std::memory_order_release);
                break;
            }
            if (args.quiet || now - lastLine < STATUS_INTERVAL) { continue; }
            const double dt = Seconds{now - lastLine}.count();
            lastLine = now;
            std::string line;
            for (size_t i = 0; i < stats.size(); ++i) {
                const uint64_t bytes = stats[i]->bytes.load();
                const uint64_t flops = stats[i]->flops.load();
                line += std::format("{}[{}] ", i == 0 ? "" : "   ", selected[i]);
                if (stats[i]->done.load() && !stats[i]->error.empty()) {
                    line += "failed";
                } else if (!stats[i]->ready.load()) {
                    line += "setting up";
                } else {
                    line += formatRates(static_cast<double>(bytes - last[i].first) / dt,
                                        static_cast<double>(flops - last[i].second) / dt);
                    if (args.periodic()) { line += std::format(" @{:3}%", stats[i]->dutyNow.load()); }
                }
                last[i] = {bytes, flops};
            }
            const std::lock_guard lock{g_print};
            std::cout << '\r' << line << "\x1b[K" << std::flush;
        }
        g_stop.store(true, std::memory_order_release);

        for (auto &w : workers) { w.join(); }

        const double elapsed = Seconds{Clock::now() - start}.count();
        int status = 0;
        std::cout << std::format("\nStopped after {:.1f} s.\n", elapsed);
        for (size_t i = 0; i < stats.size(); ++i) {
            const auto &s = *stats[i];
            if (!s.error.empty()) {
                std::cerr << std::format("[{}] Error: {}\n", selected[i], s.error);
                status = 1;
                continue;
            }
            std::cout << std::format("[{}] {} dispatches, average {}\n", selected[i], s.dispatches.load(),
                                     formatRates(static_cast<double>(s.bytes.load()) / elapsed,
                                                 static_cast<double>(s.flops.load()) / elapsed));
        }
        return status;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
