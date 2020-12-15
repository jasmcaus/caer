#include <gtest/gtest.h>

#include <ATen/ATen.h>

using namespace at;

std::vector<std::vector<int64_t>> sizes = {{1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {3, 1, 2}, {3, 2, 1}, {2, 3, 1}};

TEST(MemoryOverlapTest, TensorExpanded) {
  for (auto size : sizes) {
    Tensor t = at::ones({1}).expand(size);
    EXPECT_FALSE(t.is_contiguous());
    EXPECT_FALSE(t.is_non_overlapping_and_dense());
  }
}

TEST(MemoryOverlapTest, ScalarExpanded) {
  for (auto size : sizes) {
    Tensor t = at::tensor(1).expand(size);
    EXPECT_FALSE(t.is_contiguous());
    EXPECT_FALSE(t.is_non_overlapping_and_dense());
  }
}

TEST(MemoryOverlapTest, NonContiguousTensor) {
  for (auto size : sizes) {
    Tensor t = at::rand(size).transpose(1, 2).transpose(0, 2);
    if (!t.is_contiguous()) {
      EXPECT_TRUE(t.is_non_overlapping_and_dense());
    }
  }
}

TEST(MemoryOverlapTest, NonContiguousExpandedTensor) {
  for (auto size : sizes) {
    Tensor t = at::rand(size).transpose(1, 2).transpose(0, 2);
    if (!t.is_contiguous()) {
      for (auto size_to_add : {1, 2, 3, 4}) {
        auto transpose_size = t.sizes().vec();
        std::vector<int64_t> expanded_size(transpose_size);
        expanded_size.insert(expanded_size.begin(), size_to_add);
        auto expanded = t.expand(expanded_size);
        EXPECT_FALSE(t.is_contiguous());
        if (size_to_add == 1) {
          EXPECT_TRUE(expanded.is_non_overlapping_and_dense());
        } else {
          EXPECT_FALSE(expanded.is_non_overlapping_and_dense());
        }
      }
    }
  }
}

TEST(MemoryOverlapTest, ContiguousTensor) {
  for (auto size : sizes) {
    Tensor t = at::rand(size);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_TRUE(t.is_non_overlapping_and_dense());
  }
}

TEST(MemoryOverlapTest, ContiguousExpandedTensor) {
  for (auto size : sizes) {
    Tensor t = at::rand(size);
    for (auto size_to_add : {1, 2, 3, 4}) {
      std::vector<int64_t> expanded_size(size);
      expanded_size.insert(expanded_size.begin(), size_to_add);
      auto expanded = t.expand(expanded_size);
      EXPECT_TRUE(t.is_contiguous());
      EXPECT_TRUE(t.is_non_overlapping_and_dense());
    }
  }
}
