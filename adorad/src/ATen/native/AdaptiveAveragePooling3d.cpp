#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

namespace at {
namespace native {

namespace {

inline int start_index(int a, int b, int c) {
  return (int)std::floor((float)(a * c) / b);
}

inline int end_index(int a, int b, int c) {
  return (int)std::ceil((float)((a + 1) * c) / b);
}

template <typename scalar_t>
static void adaptive_avg_pool3d_out_frame(
    scalar_t* input_p,
    scalar_t* output_p,
    int64_t sizeD,
    int64_t isizeT,
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeT,
    int64_t osizeH,
    int64_t osizeW,
    int64_t istrideD,
    int64_t istrideT,
    int64_t istrideH,
    int64_t istrideW) {
  int64_t d = 0;
  at::internal::lazy_init_num_threads();
#pragma omp parallel for private(d)
  for (d = 0; d < sizeD; d++) {
    /* loop over output */
    int64_t ot, oh, ow;
    for (ot = 0; ot < osizeT; ot++) {
      int istartT = start_index(ot, osizeT, isizeT);
      int iendT = end_index(ot, osizeT, isizeT);
      int kT = iendT - istartT;

      for (oh = 0; oh < osizeH; oh++) {
        int istartH = start_index(oh, osizeH, isizeH);
        int iendH = end_index(oh, osizeH, isizeH);
        int kH = iendH - istartH;

        for (ow = 0; ow < osizeW; ow++) {
          int istartW = start_index(ow, osizeW, isizeW);
          int iendW = end_index(ow, osizeW, isizeW);
          int kW = iendW - istartW;

          /* local pointers */
          scalar_t* ip = input_p + d * istrideD + istartT * istrideT +
              istartH * istrideH + istartW * istrideW;
          scalar_t* op = output_p + d * osizeT * osizeH * osizeW +
              ot * osizeH * osizeW + oh * osizeW + ow;

          /* compute local average: */
          scalar_t sum = 0;
          int it, ih, iw;
          for (it = 0; it < kT; it++) {
            for (ih = 0; ih < kH; ih++) {
              for (iw = 0; iw < kW; iw++) {
                scalar_t val =
                    *(ip + it * istrideT + ih * istrideH + iw * istrideW);
                sum += val;
              }
            }
          }

          /* set output to local average */
          *op = sum / kT / kH / kW;
        }
      }
    }
  }
}

void adaptive_avg_pool3d_out_cpu_template(
    Tensor& output,
    Tensor const& input,
    IntArrayRef output_size) {
  TORCH_CHECK(output_size.size() == 3, "adaptive_avg_pool3d: output_size must be 3");
  for (int64_t i = 0; i < input.ndimension(); i++) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_avg_pool3d(): expected input to have non-empty spatial dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  /* sizes */
  int64_t sizeD = input.size(-4);
  int64_t isizeT = input.size(-3);
  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);
  /* strides */
  int64_t istrideD = input.stride(-4);
  int64_t istrideT = input.stride(-3);
  int64_t istrideH = input.stride(-2);
  int64_t istrideW = input.stride(-1);
  /* output sizes */
  auto osizeT = output_size[0];
  auto osizeH = output_size[1];
  auto osizeW = output_size[2];

  if (input.ndimension() == 4) {
    output.resize_({sizeD, osizeT, osizeH, osizeW});

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "adaptive_avg_pool3d_cpu", [&] {
          auto input_data = input.data_ptr<scalar_t>();
          auto output_data = output.data_ptr<scalar_t>();
          adaptive_avg_pool3d_out_frame<scalar_t>(
              input_data,
              output_data,
              sizeD,
              isizeT,
              isizeH,
              isizeW,
              osizeT,
              osizeH,
              osizeW,
              istrideD,
              istrideT,
              istrideH,
              istrideW);
        });
  } else {
    output.resize_({input.size(-5), sizeD, osizeT, osizeH, osizeW});
    at::internal::lazy_init_num_threads();
    int64_t b = 0;
#pragma omp parallel for private(b)
    for (b = 0; b < input.size(0); b++) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          input.scalar_type(), "adaptive_avg_pool3d_cpu", [&] {
            auto input_data = input.data_ptr<scalar_t>();
            auto output_data = output.data_ptr<scalar_t>();
            adaptive_avg_pool3d_out_frame<scalar_t>(
                input_data + b * input.stride(0),
                output_data + b * sizeD * osizeT * osizeH * osizeW,
                sizeD,
                isizeT,
                isizeH,
                isizeW,
                osizeT,
                osizeH,
                osizeW,
                istrideD,
                istrideT,
                istrideH,
                istrideW);
          });
    }
  }
}

template <typename scalar_t>
static void adaptive_avg_pool3d_backward_out_frame(
    scalar_t* gradInput_p,
    scalar_t* gradOutput_p,
    int64_t sizeD,
    int64_t isizeT,
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeT,
    int64_t osizeH,
    int64_t osizeW) {
  int64_t d = 0;
  at::internal::lazy_init_num_threads();
#pragma omp parallel for private(d)
  for (d = 0; d < sizeD; d++) {
    scalar_t* gradInput_p_d = gradInput_p + d * isizeT * isizeW * isizeH;
    scalar_t* gradOutput_p_d = gradOutput_p + d * osizeT * osizeW * osizeH;

    /* calculate average */
    int64_t ot, oh, ow;
    for (ot = 0; ot < osizeT; ot++) {
      int istartT = start_index(ot, osizeT, isizeT);
      int iendT = end_index(ot, osizeT, isizeT);
      int kT = iendT - istartT;

      for (oh = 0; oh < osizeH; oh++) {
        int istartH = start_index(oh, osizeH, isizeH);
        int iendH = end_index(oh, osizeH, isizeH);
        int kH = iendH - istartH;

        for (ow = 0; ow < osizeW; ow++) {
          int istartW = start_index(ow, osizeW, isizeW);
          int iendW = end_index(ow, osizeW, isizeW);
          int kW = iendW - istartW;

          scalar_t grad_delta =
              gradOutput_p_d[ot * osizeH * osizeW + oh * osizeW + ow] / kT /
              kH / kW;

          int it, ih, iw;
          for (it = istartT; it < iendT; it++) {
            for (ih = istartH; ih < iendH; ih++) {
              for (iw = istartW; iw < iendW; iw++) {
                /* update gradient */
                gradInput_p_d[it * isizeH * isizeW + ih * isizeW + iw] +=
                    grad_delta;
              }
            }
          }
        }
      }
    }
  }
}

Tensor& adaptive_avg_pool3d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input) {
  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();

  /* sizes */
  int64_t sizeD = input.size(-4);
  int64_t isizeT = input.size(-3);
  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);
  int64_t osizeT = gradOutput.size(-3);
  int64_t osizeH = gradOutput.size(-2);
  int64_t osizeW = gradOutput.size(-1);

  /* backprop */
  if (input.ndimension() == 4) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "adaptive_avg_pool3d_backward_cpu", [&] {
          /* get raw pointers */
          scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
          scalar_t* gradOutput_data = gradOutput.data_ptr<scalar_t>();

          adaptive_avg_pool3d_backward_out_frame<scalar_t>(
              gradInput_data,
              gradOutput_data,
              sizeD,
              isizeT,
              isizeH,
              isizeW,
              osizeT,
              osizeH,
              osizeW);
        });
  } else {
    at::internal::lazy_init_num_threads();
    int64_t b = 0;
#pragma omp parallel for private(b)
    for (b = 0; b < input.size(0); b++) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          input.scalar_type(), "adaptive_avg_pool3d_backward_cpu", [&] {
            /* get raw pointers */
            scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
            scalar_t* gradOutput_data = gradOutput.data_ptr<scalar_t>();
            adaptive_avg_pool3d_backward_out_frame<scalar_t>(
                gradInput_data + b * sizeD * isizeT * isizeH * isizeW,
                gradOutput_data + b * sizeD * osizeT * osizeH * osizeW,
                sizeD,
                isizeT,
                isizeH,
                isizeW,
                osizeT,
                osizeH,
                osizeW);
          });
    }
  }
  return gradInput;
}

} // namespace

Tensor& adaptive_avg_pool3d_out_cpu(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  adaptive_avg_pool3d_out_cpu_template(output, input, output_size);
  return output;
}

Tensor adaptive_avg_pool3d_cpu(Tensor const& input, IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  adaptive_avg_pool3d_out_cpu_template(output, input, output_size);
  return output;
}

Tensor& adaptive_avg_pool3d_backward_out_cpu(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input) {
  gradInput.resize_as_(input).zero_();
  adaptive_avg_pool3d_backward_out_cpu_template(gradInput, gradOutput_, input);
  return gradInput;
}

Tensor adaptive_avg_pool3d_backward_cpu(
    const Tensor& gradOutput_,
    const Tensor& input) {
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  adaptive_avg_pool3d_backward_out_cpu_template(gradInput, gradOutput_, input);
  return gradInput;
}

} // namespace native
} // namespace at
