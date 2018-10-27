#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

typedef float float4_t __attribute__((ext_vector_type(4)));

#define HIP_CHECK(status)                                                      \
  if (status != hipSuccess)                                                    \
    std::cout << "Got: " << hipGetErrorString(status) << " at: " << __LINE__   \
              << std::endl;

constexpr int k_elements_workitem = 4;
constexpr int k_num_workgroups = 1;
constexpr int k_iter = 32;

constexpr size_t k_size = 1024 * 1024;
constexpr int k_length = k_size / 4;

std::vector<std::string> v_file_names = {
    "copy_256_16_16_256.co", "copy_256_8_32_256.co", "copy_256_4_64_256.co",
    "copy_256_2_128_256.co",

    "copy_128_16_8_512.co",  "copy_128_8_16_512.co", "copy_128_4_32_512.co",
    "copy_128_2_64_512.co",

    "copy_64_8_8_1024.co",   "copy_64_4_16_1024.co", "copy_64_2_32_1024.co",

};

std::vector<std::string> v_kernel_names = {
    "foo_256_16_16_256", "foo_256_8_32_256", "foo_256_4_64_256",
    "foo_256_2_128_256",

    "foo_128_16_8_512",  "foo_128_8_16_512", "foo_128_4_32_512",
    "foo_128_2_64_512",

    "foo_64_8_8_1024",   "foo_64_4_16_1024", "foo_64_2_32_1024"};

std::vector<int> v_total_loops = {256, 256, 256, 256, 128, 128,
                                  128, 128, 64,  64,  64};

std::vector<int> v_unroll_factor = {16, 8, 4, 2, 16, 8, 4, 2, 8, 4, 2};

std::vector<int> v_num_loops = {16, 32, 64, 128, 8, 16, 32, 64, 8, 16, 32};

std::vector<int> v_num_workitems = {256, 256, 256,  256,  512, 512,
                                    512, 512, 1024, 1024, 1024};

template <int blockSize, int numLoops>
__global__ void Foo16(float4_t *p_in, float4_t *p_out) {
  int tx = threadIdx.x;
  for (int i = 0; i < numLoops; i++) {
#pragma unroll 16
    for (int j = 0; j < 16; j++) {
      int index = tx + j * blockSize + i * 16 * blockSize;
      p_out[index] = p_in[index];
    }
  }
}

template <int blockSize, int numLoops>
__global__ void Foo8(float4_t *p_in, float4_t *p_out) {
  int tx = threadIdx.x;
  for (int i = 0; i < numLoops; i++) {
#pragma unroll 8
    for (int j = 0; j < 8; j++) {
      int index = tx + j * blockSize + i * 8 * blockSize;
      p_out[index] = p_in[index];
    }
  }
}

template <int blockSize, int numLoops>
__global__ void Foo4(float4_t *p_in, float4_t *p_out) {
  int tx = threadIdx.x;
  for (int i = 0; i < numLoops; i++) {
#pragma unroll 4
    for (int j = 0; j < 4; j++) {
      int index = tx + j * blockSize + i * 4 * blockSize;
      p_out[index] = p_in[index];
    }
  }
}

template <int blockSize, int numLoops>
__global__ void Foo2(float4_t *p_in, float4_t *p_out) {
  int tx = threadIdx.x;
  for (int i = 0; i < numLoops; i++) {
#pragma unroll 2
    for (int j = 0; j < 2; j++) {
      int index = tx + j * blockSize + i * 2 * blockSize;
      p_out[index] = p_in[index];
    }
  }
}

int gen() {
  static int i = 0;
  return ++i;
}

static bool enable_benchmarking_mode = false;

int main(int argc, char *argv[]) {
  if (argc == 2) {
    if (strcmp(argv[1], "-b") == 0) {
      enable_benchmarking_mode = true;
    }
  }
  std::vector<float> h_input(k_length);
  std::vector<float> h_output(k_length);

  std::cout << "Length: " << k_length << std::endl;

  std::generate(h_input.begin(), h_input.end(), gen);

  float *d_input, *d_output;

  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  std::vector<hipModule_t> v_modules(v_file_names.size());
  std::vector<hipFunction_t> v_functions(v_kernel_names.size());
  //  hipModule_t* v_modules = new hipModule_t[v_file_names.size()];
  //  hipFunction_t* v_functions = new hipFunction_t[v_file_names.size()];
  for (size_t i = 0; i < v_file_names.size(); i++) {
    if (!enable_benchmarking_mode)
      std::cout << "Loading file " << v_file_names[i] << " ... " << std::endl;
    HIP_CHECK(hipModuleLoad(&v_modules[i], v_file_names[i].c_str()));
  }

  for (size_t i = 0; i < v_kernel_names.size(); i++) {
    if (!enable_benchmarking_mode)
      std::cout << "Finding kernel " << v_kernel_names[i] << " ... "
                << std::endl;
    HIP_CHECK(hipModuleGetFunction(&v_functions[i], v_modules[i],
                                   v_kernel_names[i].c_str()));
  }

  HIP_CHECK(hipMalloc(&d_input, k_size));
  HIP_CHECK(hipMalloc(&d_output, k_size));
  HIP_CHECK(hipMemcpy(d_input, h_input.data(), k_size, hipMemcpyHostToDevice));

  hipEvent_t event;
  hipStream_t stream;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipEventCreateWithFlags(&event, hipEventReleaseToSystem));

  if (!enable_benchmarking_mode)
    std::cout << d_input << " " << d_output << std::endl;

  struct {
    void *input;
    void *output;
  } args;

  size_t size_args = sizeof(args);

  std::generate(h_input.begin(), h_input.end(), gen);
  args.input = d_input;

  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size_args,
                    HIP_LAUNCH_PARAM_END};

  if (enable_benchmarking_mode)
    std::cout << "Kernel Name\t\tTotal Loops\t\tUnroll Factor\t\tNum "
                 "Loops\t\tBandwidth"
              << std::endl;

  for (size_t i = 0; i < v_kernel_names.size(); i++) {
    if (!enable_benchmarking_mode)
      std::cout << "\033[1;34mRunning kernel " << v_kernel_names[i]
                << " ... \033[0m" << std::endl;
    std::fill(h_output.begin(), h_output.end(), -1.0f);

    HIP_CHECK(
        hipMemcpy(d_output, h_output.data(), k_size, hipMemcpyHostToDevice));

    args.output = d_output;

    HIP_CHECK(hipModuleLaunchKernel(v_functions[i], k_num_workgroups, 1, 1,
                                    v_num_workitems[i], 1, 1, 0, stream, NULL,
                                    (void **)&config));
    HIP_CHECK(hipEventRecord(event, stream));
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < k_iter; j++) {
      HIP_CHECK(hipModuleLaunchKernel(v_functions[i], k_num_workgroups, 1, 1,
                                      v_num_workitems[i], 1, 1, 0, stream, NULL,
                                      (void **)&config));
      HIP_CHECK(hipEventRecord(event, stream));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();

    HIP_CHECK(
        hipMemcpy(h_output.data(), d_output, k_size, hipMemcpyDeviceToHost));

    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count();
    double bytes = 2 * k_size * k_iter;
    double bw = bytes / sec / double(1000 * 1000 * 1000);

    if (!enable_benchmarking_mode) {

      std::cout << "Total Time taken: " << sec << std::endl;
      std::cout << "Time taken per iteration: " << double(sec) / double(k_iter)
                << std::endl;

      std::cout << "\033[1;32mtotal loops = " << v_total_loops[i]
                << " unroll factor = " << v_unroll_factor[i]
                << " num loops = " << v_num_loops[i] << "\033[0m" << std::endl;
      std::cout << "\033[1;32mBandwidth: " << bw << " GBps\033[0m" << std::endl;

    } else {
      std::cout << v_kernel_names[i] + "_asm"
                << "\t\t" << v_total_loops[i] << "\t\t" << v_unroll_factor[i]
                << "\t\t\t" << v_num_loops[i] << "\t\t\t" << bw << std::endl;
    }

    for (int j = 0; j < k_length; j++) {
      if (h_output[j] != j + 1) {
        std::cout << "\033[0;31mBad output at: " << j << "\033[0m" << std::endl;
        break;
      }
    }

    if (!enable_benchmarking_mode)
      std::cout << "Finished running kernel " << v_kernel_names[i] << std::endl;
  }

  /**
   * Launch HIP kernels from here
   */

  {
    constexpr int k_wis = 256;
    constexpr int k_nloops = 16;
    constexpr int k_tloops = 256;
    std::string kernel_name = "foo_" + std::to_string(k_tloops) + "_" +
                              std::to_string(k_tloops / k_nloops) + "_" +
                              std::to_string(k_nloops) + "_" +
                              std::to_string(k_wis) + "_hip";
    if (!enable_benchmarking_mode)
      std::cout << "\033[1;34mRunning kernel " << kernel_name << " ... \033[0m"
                << std::endl;
    std::fill(h_output.begin(), h_output.end(), -1.0f);

    HIP_CHECK(
        hipMemcpy(d_output, h_output.data(), k_size, hipMemcpyHostToDevice));

    hipLaunchKernelGGL((Foo16<k_wis, k_nloops>), dim3(1, 1, 1),
                       dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
    HIP_CHECK(hipEventRecord(event, stream));
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < k_iter; j++) {
      hipLaunchKernelGGL((Foo16<k_wis, k_nloops>), dim3(1, 1, 1),
                         dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
      HIP_CHECK(hipEventRecord(event, stream));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();

    HIP_CHECK(
        hipMemcpy(h_output.data(), d_output, k_size, hipMemcpyDeviceToHost));

    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count();
    double bytes = 2 * k_size * k_iter;
    double bw = bytes / sec / double(1000 * 1000 * 1000);

    if (!enable_benchmarking_mode) {
      std::cout << "Total Time taken: " << sec << std::endl;
      std::cout << "Time taken per iteration: " << double(sec) / double(k_iter)
                << std::endl;

      std::cout << "\033[1;32mtotal loops = " << k_tloops
                << " unroll factor = " << k_tloops / k_nloops
                << " num loops = " << k_nloops << "\033[0m" << std::endl;

      std::cout << "\033[1;32mBandwidth: " << bw << " GBps\033[0m" << std::endl;
    }

    for (int j = 0; j < k_length; j++) {
      if (h_output[j] != j + 1) {
        std::cout << "\033[0;31mBad output at: " << j << "\033[0m" << std::endl;
        break;
      }
    }

    if (enable_benchmarking_mode) {
      std::cout << kernel_name << "\t\t" << k_tloops << "\t\t"
                << k_tloops / k_nloops << "\t\t\t" << k_nloops << "\t\t\t" << bw
                << std::endl;
    } else {
      std::cout << "Finished running kernel " << kernel_name << std::endl;
    }
  }

  {
    constexpr int k_wis = 256;
    constexpr int k_nloops = 32;
    constexpr int k_tloops = 256;
    std::string kernel_name = "foo_" + std::to_string(k_tloops) + "_" +
                              std::to_string(k_tloops / k_nloops) + "_" +
                              std::to_string(k_nloops) + "_" +
                              std::to_string(k_wis) + "_hip";

    if (!enable_benchmarking_mode)
      std::cout << "\033[1;34mRunning kernel " << kernel_name << " ... \033[0m"
                << std::endl;

    std::fill(h_output.begin(), h_output.end(), -1.0f);

    HIP_CHECK(
        hipMemcpy(d_output, h_output.data(), k_size, hipMemcpyHostToDevice));

    hipLaunchKernelGGL((Foo8<k_wis, k_nloops>), dim3(1, 1, 1),
                       dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
    HIP_CHECK(hipEventRecord(event, stream));
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < k_iter; j++) {
      hipLaunchKernelGGL((Foo8<k_wis, k_nloops>), dim3(1, 1, 1),
                         dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
      HIP_CHECK(hipEventRecord(event, stream));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();

    HIP_CHECK(
        hipMemcpy(h_output.data(), d_output, k_size, hipMemcpyDeviceToHost));

    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count();
    double bytes = 2 * k_size * k_iter;
    double bw = bytes / sec / double(1000 * 1000 * 1000);

    if (!enable_benchmarking_mode) {
      std::cout << "Total Time taken: " << sec << std::endl;
      std::cout << "Time taken per iteration: " << double(sec) / double(k_iter)
                << std::endl;
      std::cout << "\033[1;32mtotal loops = " << k_tloops
                << " unroll factor = " << k_tloops / k_nloops
                << " num loops = " << k_nloops << "\033[0m" << std::endl;

      std::cout << "\033[1;32mBandwidth: " << bw << " GBps\033[0m" << std::endl;
    }

    for (int j = 0; j < k_length; j++) {
      if (h_output[j] != j + 1) {
        std::cout << "\033[0;31mBad output at: " << j << "\033[0m" << std::endl;
        break;
      }
    }

    if (enable_benchmarking_mode) {
      std::cout << kernel_name << "\t\t" << k_tloops << "\t\t"
                << k_tloops / k_nloops << "\t\t\t" << k_nloops << "\t\t\t" << bw
                << std::endl;
    } else {
      std::cout << "Finished running kernel " << kernel_name << std::endl;
    }
  }

  {
    constexpr int k_wis = 256;
    constexpr int k_nloops = 64;
    constexpr int k_tloops = 256;
    std::string kernel_name = "foo_" + std::to_string(k_tloops) + "_" +
                              std::to_string(k_tloops / k_nloops) + "_" +
                              std::to_string(k_nloops) + "_" +
                              std::to_string(k_wis) + "_hip";

    if (!enable_benchmarking_mode)
      std::cout << "\033[1;34mRunning kernel " << kernel_name << " ... \033[0m"
                << std::endl;

    std::fill(h_output.begin(), h_output.end(), -1.0f);

    HIP_CHECK(
        hipMemcpy(d_output, h_output.data(), k_size, hipMemcpyHostToDevice));

    hipLaunchKernelGGL((Foo4<k_wis, k_nloops>), dim3(1, 1, 1),
                       dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
    HIP_CHECK(hipEventRecord(event, stream));
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < k_iter; j++) {
      hipLaunchKernelGGL((Foo4<k_wis, k_nloops>), dim3(1, 1, 1),
                         dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
      HIP_CHECK(hipEventRecord(event, stream));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();

    HIP_CHECK(
        hipMemcpy(h_output.data(), d_output, k_size, hipMemcpyDeviceToHost));

    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count();
    double bytes = 2 * k_size * k_iter;
    double bw = bytes / sec / double(1000 * 1000 * 1000);

    if (!enable_benchmarking_mode) {
      std::cout << "Total Time taken: " << sec << std::endl;
      std::cout << "Time taken per iteration: " << double(sec) / double(k_iter)
                << std::endl;
      std::cout << "\033[1;32mtotal loops = " << k_tloops
                << " unroll factor = " << k_tloops / k_nloops
                << " num loops = " << k_nloops << "\033[0m" << std::endl;

      std::cout << "\033[1;32mBandwidth: " << bw << " GBps\033[0m" << std::endl;
    }

    for (int j = 0; j < k_length; j++) {
      if (h_output[j] != j + 1) {
        std::cout << "\033[0;31mBad output at: " << j << "\033[0m" << std::endl;
        break;
      }
    }

    if (enable_benchmarking_mode) {
      std::cout << kernel_name << "\t\t" << k_tloops << "\t\t"
                << k_tloops / k_nloops << "\t\t\t" << k_nloops << "\t\t\t" << bw
                << std::endl;
    } else {
      std::cout << "Finished running kernel " << kernel_name << std::endl;
    }
  }

  {
    constexpr int k_wis = 256;
    constexpr int k_nloops = 128;
    constexpr int k_tloops = 256;
    std::string kernel_name = "foo_" + std::to_string(k_tloops) + "_" +
                              std::to_string(k_tloops / k_nloops) + "_" +
                              std::to_string(k_nloops) + "_" +
                              std::to_string(k_wis) + "_hip";

    if (!enable_benchmarking_mode)
      std::cout << "\033[1;34mRunning kernel " << kernel_name << " ... \033[0m"
                << std::endl;

    std::fill(h_output.begin(), h_output.end(), -1.0f);

    HIP_CHECK(
        hipMemcpy(d_output, h_output.data(), k_size, hipMemcpyHostToDevice));

    hipLaunchKernelGGL((Foo2<k_wis, k_nloops>), dim3(1, 1, 1),
                       dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
    HIP_CHECK(hipEventRecord(event, stream));
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < k_iter; j++) {
      hipLaunchKernelGGL((Foo2<k_wis, k_nloops>), dim3(1, 1, 1),
                         dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
      HIP_CHECK(hipEventRecord(event, stream));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();

    HIP_CHECK(
        hipMemcpy(h_output.data(), d_output, k_size, hipMemcpyDeviceToHost));

    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count();
    double bytes = 2 * k_size * k_iter;
    double bw = bytes / sec / double(1000 * 1000 * 1000);

    if (!enable_benchmarking_mode) {
      std::cout << "Total Time taken: " << sec << std::endl;
      std::cout << "Time taken per iteration: " << double(sec) / double(k_iter)
                << std::endl;
      std::cout << "\033[1;32mtotal loops = " << k_tloops
                << " unroll factor = " << k_tloops / k_nloops
                << " num loops = " << k_nloops << "\033[0m" << std::endl;

      std::cout << "\033[1;32mBandwidth: " << bw << " GBps\033[0m" << std::endl;
    }

    for (int j = 0; j < k_length; j++) {
      if (h_output[j] != j + 1) {
        std::cout << "\033[0;31mBad output at: " << j << "\033[0m" << std::endl;
        break;
      }
    }

    if (enable_benchmarking_mode) {
      std::cout << kernel_name << "\t\t" << k_tloops << "\t\t"
                << k_tloops / k_nloops << "\t\t\t" << k_nloops << "\t\t\t" << bw
                << std::endl;
    } else {
      std::cout << "Finished running kernel " << kernel_name << std::endl;
    }
  }

  {
    constexpr int k_wis = 512;
    constexpr int k_nloops = 8;
    constexpr int k_tloops = 128;
    std::string kernel_name = "foo_" + std::to_string(k_tloops) + "_" +
                              std::to_string(k_tloops / k_nloops) + "_" +
                              std::to_string(k_nloops) + "_" +
                              std::to_string(k_wis) + "_hip";

    if (!enable_benchmarking_mode)
      std::cout << "\033[1;34mRunning kernel " << kernel_name << " ... \033[0m"
                << std::endl;

    std::fill(h_output.begin(), h_output.end(), -1.0f);

    HIP_CHECK(
        hipMemcpy(d_output, h_output.data(), k_size, hipMemcpyHostToDevice));

    hipLaunchKernelGGL((Foo16<k_wis, k_nloops>), dim3(1, 1, 1),
                       dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
    HIP_CHECK(hipEventRecord(event, stream));
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < k_iter; j++) {
      hipLaunchKernelGGL((Foo16<k_wis, k_nloops>), dim3(1, 1, 1),
                         dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
      HIP_CHECK(hipEventRecord(event, stream));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();

    HIP_CHECK(
        hipMemcpy(h_output.data(), d_output, k_size, hipMemcpyDeviceToHost));

    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count();
    double bytes = 2 * k_size * k_iter;
    double bw = bytes / sec / double(1000 * 1000 * 1000);

    if (!enable_benchmarking_mode) {
      std::cout << "Total Time taken: " << sec << std::endl;
      std::cout << "Time taken per iteration: " << double(sec) / double(k_iter)
                << std::endl;
      std::cout << "\033[1;32mtotal loops = " << k_tloops
                << " unroll factor = " << k_tloops / k_nloops
                << " num loops = " << k_nloops << "\033[0m" << std::endl;

      std::cout << "\033[1;32mBandwidth: " << bw << " GBps\033[0m" << std::endl;
    }

    for (int j = 0; j < k_length; j++) {
      if (h_output[j] != j + 1) {
        std::cout << "\033[0;31mBad output at: " << j << "\033[0m" << std::endl;
        break;
      }
    }

    if (enable_benchmarking_mode) {
      std::cout << kernel_name << "\t\t" << k_tloops << "\t\t"
                << k_tloops / k_nloops << "\t\t\t" << k_nloops << "\t\t\t" << bw
                << std::endl;
    } else {
      std::cout << "Finished running kernel " << kernel_name << std::endl;
    }
  }

  {
    constexpr int k_wis = 512;
    constexpr int k_nloops = 16;
    constexpr int k_tloops = 128;
    std::string kernel_name = "foo_" + std::to_string(k_tloops) + "_" +
                              std::to_string(k_tloops / k_nloops) + "_" +
                              std::to_string(k_nloops) + "_" +
                              std::to_string(k_wis) + "_hip";

    if (!enable_benchmarking_mode)
      std::cout << "\033[1;34mRunning kernel " << kernel_name << " ... \033[0m"
                << std::endl;

    std::fill(h_output.begin(), h_output.end(), -1.0f);

    HIP_CHECK(
        hipMemcpy(d_output, h_output.data(), k_size, hipMemcpyHostToDevice));

    hipLaunchKernelGGL((Foo8<k_wis, k_nloops>), dim3(1, 1, 1),
                       dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
    HIP_CHECK(hipEventRecord(event, stream));
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < k_iter; j++) {
      hipLaunchKernelGGL((Foo8<k_wis, k_nloops>), dim3(1, 1, 1),
                         dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
      HIP_CHECK(hipEventRecord(event, stream));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();

    HIP_CHECK(
        hipMemcpy(h_output.data(), d_output, k_size, hipMemcpyDeviceToHost));

    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count();
    double bytes = 2 * k_size * k_iter;
    double bw = bytes / sec / double(1000 * 1000 * 1000);

    if (!enable_benchmarking_mode) {
      std::cout << "Total Time taken: " << sec << std::endl;
      std::cout << "Time taken per iteration: " << double(sec) / double(k_iter)
                << std::endl;
      std::cout << "\033[1;32mtotal loops = " << k_tloops
                << " unroll factor = " << k_tloops / k_nloops
                << " num loops = " << k_nloops << "\033[0m" << std::endl;

      std::cout << "\033[1;32mBandwidth: " << bw << " GBps\033[0m" << std::endl;
    }

    for (int j = 0; j < k_length; j++) {
      if (h_output[j] != j + 1) {
        std::cout << "\033[0;31mBad output at: " << j << "\033[0m" << std::endl;
        break;
      }
    }

    if (enable_benchmarking_mode) {
      std::cout << kernel_name << "\t\t" << k_tloops << "\t\t"
                << k_tloops / k_nloops << "\t\t\t" << k_nloops << "\t\t\t" << bw
                << std::endl;
    } else {
      std::cout << "Finished running kernel " << kernel_name << std::endl;
    }
  }

  {
    constexpr int k_wis = 512;
    constexpr int k_nloops = 32;
    constexpr int k_tloops = 128;
    std::string kernel_name = "foo_" + std::to_string(k_tloops) + "_" +
                              std::to_string(k_tloops / k_nloops) + "_" +
                              std::to_string(k_nloops) + "_" +
                              std::to_string(k_wis) + "_hip";

    if (!enable_benchmarking_mode)
      std::cout << "\033[1;34mRunning kernel " << kernel_name << " ... \033[0m"
                << std::endl;

    std::fill(h_output.begin(), h_output.end(), -1.0f);

    HIP_CHECK(
        hipMemcpy(d_output, h_output.data(), k_size, hipMemcpyHostToDevice));

    hipLaunchKernelGGL((Foo4<k_wis, k_nloops>), dim3(1, 1, 1),
                       dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
    HIP_CHECK(hipEventRecord(event, stream));
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < k_iter; j++) {
      hipLaunchKernelGGL((Foo4<k_wis, k_nloops>), dim3(1, 1, 1),
                         dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
      HIP_CHECK(hipEventRecord(event, stream));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();

    HIP_CHECK(
        hipMemcpy(h_output.data(), d_output, k_size, hipMemcpyDeviceToHost));

    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count();
    double bytes = 2 * k_size * k_iter;
    double bw = bytes / sec / double(1000 * 1000 * 1000);

    if (!enable_benchmarking_mode) {
      std::cout << "Total Time taken: " << sec << std::endl;
      std::cout << "Time taken per iteration: " << double(sec) / double(k_iter)
                << std::endl;
      std::cout << "\033[1;32mtotal loops = " << k_tloops
                << " unroll factor = " << k_tloops / k_nloops
                << " num loops = " << k_nloops << "\033[0m" << std::endl;

      std::cout << "\033[1;32mBandwidth: " << bw << " GBps\033[0m" << std::endl;
    }

    for (int j = 0; j < k_length; j++) {
      if (h_output[j] != j + 1) {
        std::cout << "\033[0;31mBad output at: " << j << "\033[0m" << std::endl;
        break;
      }
    }

    if (enable_benchmarking_mode) {
      std::cout << kernel_name << "\t\t" << k_tloops << "\t\t"
                << k_tloops / k_nloops << "\t\t\t" << k_nloops << "\t\t\t" << bw
                << std::endl;
    } else {
      std::cout << "Finished running kernel " << kernel_name << std::endl;
    }
  }

  {
    constexpr int k_wis = 512;
    constexpr int k_nloops = 64;
    constexpr int k_tloops = 128;
    std::string kernel_name = "foo_" + std::to_string(k_tloops) + "_" +
                              std::to_string(k_tloops / k_nloops) + "_" +
                              std::to_string(k_nloops) + "_" +
                              std::to_string(k_wis) + "_hip";

    if (!enable_benchmarking_mode)
      std::cout << "\033[1;34mRunning kernel " << kernel_name << " ... \033[0m"
                << std::endl;

    std::fill(h_output.begin(), h_output.end(), -1.0f);

    HIP_CHECK(
        hipMemcpy(d_output, h_output.data(), k_size, hipMemcpyHostToDevice));

    hipLaunchKernelGGL((Foo2<k_wis, k_nloops>), dim3(1, 1, 1),
                       dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
    HIP_CHECK(hipEventRecord(event, stream));
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < k_iter; j++) {
      hipLaunchKernelGGL((Foo2<k_wis, k_nloops>), dim3(1, 1, 1),
                         dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
      HIP_CHECK(hipEventRecord(event, stream));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();

    HIP_CHECK(
        hipMemcpy(h_output.data(), d_output, k_size, hipMemcpyDeviceToHost));

    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count();
    double bytes = 2 * k_size * k_iter;
    double bw = bytes / sec / double(1000 * 1000 * 1000);

    if (!enable_benchmarking_mode) {
      std::cout << "Total Time taken: " << sec << std::endl;
      std::cout << "Time taken per iteration: " << double(sec) / double(k_iter)
                << std::endl;
      std::cout << "\033[1;32mtotal loops = " << k_tloops
                << " unroll factor = " << k_tloops / k_nloops
                << " num loops = " << k_nloops << "\033[0m" << std::endl;

      std::cout << "\033[1;32mBandwidth: " << bw << " GBps\033[0m" << std::endl;
    }

    for (int j = 0; j < k_length; j++) {
      if (h_output[j] != j + 1) {
        std::cout << "\033[0;31mBad output at: " << j << "\033[0m" << std::endl;
        break;
      }
    }

    if (enable_benchmarking_mode) {
      std::cout << kernel_name << "\t\t" << k_tloops << "\t\t"
                << k_tloops / k_nloops << "\t\t\t" << k_nloops << "\t\t\t" << bw
                << std::endl;
    } else {
      std::cout << "Finished running kernel " << kernel_name << std::endl;
    }
  }

  {
    constexpr int k_wis = 1024;
    constexpr int k_nloops = 8;
    constexpr int k_tloops = 64;
    std::string kernel_name = "foo_" + std::to_string(k_tloops) + "_" +
                              std::to_string(k_tloops / k_nloops) + "_" +
                              std::to_string(k_nloops) + "_" +
                              std::to_string(k_wis) + "_hip";

    if (!enable_benchmarking_mode)
      std::cout << "\033[1;34mRunning kernel " << kernel_name << " ... \033[0m"
                << std::endl;

    std::fill(h_output.begin(), h_output.end(), -1.0f);

    HIP_CHECK(
        hipMemcpy(d_output, h_output.data(), k_size, hipMemcpyHostToDevice));

    hipLaunchKernelGGL((Foo8<k_wis, k_nloops>), dim3(1, 1, 1),
                       dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
    HIP_CHECK(hipEventRecord(event, stream));
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < k_iter; j++) {
      hipLaunchKernelGGL((Foo8<k_wis, k_nloops>), dim3(1, 1, 1),
                         dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
      HIP_CHECK(hipEventRecord(event, stream));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();

    HIP_CHECK(
        hipMemcpy(h_output.data(), d_output, k_size, hipMemcpyDeviceToHost));

    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count();
    double bytes = 2 * k_size * k_iter;
    double bw = bytes / sec / double(1000 * 1000 * 1000);

    if (!enable_benchmarking_mode) {
      std::cout << "Total Time taken: " << sec << std::endl;
      std::cout << "Time taken per iteration: " << double(sec) / double(k_iter)
                << std::endl;
      std::cout << "\033[1;32mtotal loops = " << k_tloops
                << " unroll factor = " << k_tloops / k_nloops
                << " num loops = " << k_nloops << "\033[0m" << std::endl;

      std::cout << "\033[1;32mBandwidth: " << bw << " GBps\033[0m" << std::endl;
    }

    for (int j = 0; j < k_length; j++) {
      if (h_output[j] != j + 1) {
        std::cout << "\033[0;31mBad output at: " << j << "\033[0m" << std::endl;
        break;
      }
    }

    if (enable_benchmarking_mode) {
      std::cout << kernel_name << "\t\t" << k_tloops << "\t\t"
                << k_tloops / k_nloops << "\t\t\t" << k_nloops << "\t\t\t" << bw
                << std::endl;
    } else {
      std::cout << "Finished running kernel " << kernel_name << std::endl;
    }
  }

  {
    constexpr int k_wis = 1024;
    constexpr int k_nloops = 16;
    constexpr int k_tloops = 64;
    std::string kernel_name = "foo_" + std::to_string(k_tloops) + "_" +
                              std::to_string(k_tloops / k_nloops) + "_" +
                              std::to_string(k_nloops) + "_" +
                              std::to_string(k_wis) + "_hip";

    if (!enable_benchmarking_mode)
      std::cout << "\033[1;34mRunning kernel " << kernel_name << " ... \033[0m"
                << std::endl;

    std::fill(h_output.begin(), h_output.end(), -1.0f);

    HIP_CHECK(
        hipMemcpy(d_output, h_output.data(), k_size, hipMemcpyHostToDevice));

    hipLaunchKernelGGL((Foo4<k_wis, k_nloops>), dim3(1, 1, 1),
                       dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
    HIP_CHECK(hipEventRecord(event, stream));
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < k_iter; j++) {
      hipLaunchKernelGGL((Foo4<k_wis, k_nloops>), dim3(1, 1, 1),
                         dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
      HIP_CHECK(hipEventRecord(event, stream));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();

    HIP_CHECK(
        hipMemcpy(h_output.data(), d_output, k_size, hipMemcpyDeviceToHost));

    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count();
    double bytes = 2 * k_size * k_iter;
    double bw = bytes / sec / double(1000 * 1000 * 1000);

    if (!enable_benchmarking_mode) {
      std::cout << "Total Time taken: " << sec << std::endl;
      std::cout << "Time taken per iteration: " << double(sec) / double(k_iter)
                << std::endl;
      std::cout << "\033[1;32mtotal loops = " << k_tloops
                << " unroll factor = " << k_tloops / k_nloops
                << " num loops = " << k_nloops << "\033[0m" << std::endl;

      std::cout << "\033[1;32mBandwidth: " << bw << " GBps\033[0m" << std::endl;
    }

    for (int j = 0; j < k_length; j++) {
      if (h_output[j] != j + 1) {
        std::cout << "\033[0;31mBad output at: " << j << "\033[0m" << std::endl;
        break;
      }
    }

    if (enable_benchmarking_mode) {
      std::cout << kernel_name << "\t\t" << k_tloops << "\t\t"
                << k_tloops / k_nloops << "\t\t\t" << k_nloops << "\t\t\t" << bw
                << std::endl;
    } else {
      std::cout << "Finished running kernel " << kernel_name << std::endl;
    }
  }

  {
    constexpr int k_wis = 1024;
    constexpr int k_nloops = 32;
    constexpr int k_tloops = 64;
    std::string kernel_name = "foo_" + std::to_string(k_tloops) + "_" +
                              std::to_string(k_tloops / k_nloops) + "_" +
                              std::to_string(k_nloops) + "_" +
                              std::to_string(k_wis) + "_hip";

    if (!enable_benchmarking_mode)
      std::cout << "\033[1;34mRunning kernel " << kernel_name << " ... \033[0m"
                << std::endl;

    std::fill(h_output.begin(), h_output.end(), -1.0f);

    HIP_CHECK(
        hipMemcpy(d_output, h_output.data(), k_size, hipMemcpyHostToDevice));

    hipLaunchKernelGGL((Foo2<k_wis, k_nloops>), dim3(1, 1, 1),
                       dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
    HIP_CHECK(hipEventRecord(event, stream));
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < k_iter; j++) {
      hipLaunchKernelGGL((Foo2<k_wis, k_nloops>), dim3(1, 1, 1),
                         dim3(k_wis, 1, 1), 0, stream, d_input, d_output);
      HIP_CHECK(hipEventRecord(event, stream));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();

    HIP_CHECK(
        hipMemcpy(h_output.data(), d_output, k_size, hipMemcpyDeviceToHost));

    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count();
    double bytes = 2 * k_size * k_iter;
    double bw = bytes / sec / double(1000 * 1000 * 1000);

    if (!enable_benchmarking_mode) {
      std::cout << "Total Time taken: " << sec << std::endl;
      std::cout << "Time taken per iteration: " << double(sec) / double(k_iter)
                << std::endl;
      std::cout << "\033[1;32mtotal loops = " << k_tloops
                << " unroll factor = " << k_tloops / k_nloops
                << " num loops = " << k_nloops << "\033[0m" << std::endl;

      std::cout << "\033[1;32mBandwidth: " << bw << " GBps\033[0m" << std::endl;
    }

    for (int j = 0; j < k_length; j++) {
      if (h_output[j] != j + 1) {
        std::cout << "\033[0;31mBad output at: " << j << "\033[0m" << std::endl;
        break;
      }
    }

    if (enable_benchmarking_mode) {
      std::cout << kernel_name << "\t\t" << k_tloops << "\t\t"
                << k_tloops / k_nloops << "\t\t\t" << k_nloops << "\t\t\t" << bw
                << std::endl;
    } else {
      std::cout << "Finished running kernel " << kernel_name << std::endl;
    }
  }

  //  delete[] v_modules;
  //  delete[] v_functions;

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_output));
}
