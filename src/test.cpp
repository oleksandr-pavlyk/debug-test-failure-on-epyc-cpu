#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>
#include <iostream>
#include <string>

#include "kernels/sorting/topk.hpp"

void echo_device(const sycl::queue q)
{
  const auto &dev = q.get_device();
  const auto &name = dev.get_info<sycl::info::device::name>();
  const auto &driver_version =
    dev.get_info<sycl::info::device::driver_version>();

  std::cout << "Device: " << name << " [" << driver_version << "]\n";

  return;
}

template <typename argTy, typename IndexTy>
void run_test(sycl::queue &q, size_t n, size_t k, size_t shift) 
{
  argTy *data = sycl::malloc_device<argTy>(n, q);
  argTy *topk_vals = sycl::malloc_device<argTy>(k, q);
  IndexTy *topk_idx = sycl::malloc_device<IndexTy>(k, q);

  sycl::event pop_ev =
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for({n},[=](sycl::id<1> id) {
        const std::size_t i = id[0];
        const std::size_t shifted = ((i + shift) % n);

        data[shifted] = (2*i < n) ? argTy(1) : argTy(0);
      });
    });

  using dpctl::tensor::kernels::topk_radix_impl;
  constexpr bool ascending = false;
  constexpr std::ptrdiff_t zero{0};

  sycl::event topk_ev =
    topk_radix_impl<argTy, IndexTy>(
       q,
       std::size_t{1},
       n,
       k,
       ascending,
       reinterpret_cast<char *>(data),
       reinterpret_cast<char *>(topk_vals),
       reinterpret_cast<char *>(topk_idx),
       zero,
       zero,
       zero,
       zero,
       zero,
       zero,
       {pop_ev}
  );

  argTy *host_topk_vals = new argTy[k];
  IndexTy *host_topk_idx = new IndexTy[k];

  sycl::event copy_ev1 = q.copy<argTy>(topk_vals, host_topk_vals, k, {topk_ev});
  sycl::event copy_ev2 = q.copy<IndexTy>(topk_idx, host_topk_idx, k, {topk_ev});

  copy_ev1.wait();
  copy_ev2.wait();
  
  sycl::free(data, q);
  sycl::free(topk_vals, q);
  sycl::free(topk_idx, q);

  for(size_t i = 0; i < k; ++i) {
    std::cout << "([" << host_topk_idx[i] << "], "
	      << std::to_string(host_topk_vals[i]) << " vs "
	      << ((((host_topk_idx[i] + shift) % n)*2 < n) ? 0 : 1) << ") ";
  }
  std::cout << "\n";

  delete[] host_topk_idx;
  delete[] host_topk_vals;
}

int main(void) {
  sycl::queue q{sycl::default_selector_v};

  
  const std::size_t n = 255 * 2;
  const std::size_t k = 5;
  const std::size_t shift = 734;

  {
    std::cout << "++++++++++++ i1 i8\n";
    using argTy = std::int8_t;
    using IndexTy = std::int64_t;

    run_test<argTy, IndexTy>(q, n, k, shift);
    run_test<argTy, IndexTy>(q, n, k, shift);
    run_test<argTy, IndexTy>(q, n, k, shift);
  }

  {
    std::cout << "++++++++++++ i2 i8\n";
    using argTy = std::int16_t;
    using IndexTy = std::int64_t;

    run_test<argTy, IndexTy>(q, n, k, shift);
    run_test<argTy, IndexTy>(q, n, k, shift);
    run_test<argTy, IndexTy>(q, n, k, shift);
  }

  {
    std::cout << "++++++++++++ i1 i8\n";
    using argTy = std::int8_t;
    using IndexTy = std::int64_t;

    run_test<argTy, IndexTy>(q, n, k, shift);
    run_test<argTy, IndexTy>(q, n, k, shift);
    run_test<argTy, IndexTy>(q, n, k, shift);
  }

  return 0;
}
