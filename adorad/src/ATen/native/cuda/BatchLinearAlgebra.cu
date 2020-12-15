#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/cuda/BatchLinearAlgebraLib.h>
#include <ATen/native/cpu/zmath.h>

#include <THC/THC.h> // for USE_MAGMA

#ifdef USE_MAGMA
#include <magma.h>
#include <magma_types.h>

const bool use_magma_ = true;
#else
const bool use_magma_ = false;

#endif

namespace at {
namespace native {

#ifdef USE_MAGMA
template<class scalar_t>
void magmaSolve(
    magma_int_t n, magma_int_t nrhs, scalar_t* dA, magma_int_t ldda,
    magma_int_t* ipiv, scalar_t* dB, magma_int_t lddb, magma_int_t* info);

template<class scalar_t>
void magmaSolveBatched(
    magma_int_t n, magma_int_t nrhs, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, scalar_t** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, const MAGMAQueue& magma_queue);

template<class scalar_t>
void magmaLu(
    magma_int_t m, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info);

template<class scalar_t>
void magmaLuBatched(
    magma_int_t m, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue);

template<class scalar_t>
void magmaLuNoPiv(
    magma_int_t m, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    magma_int_t* info);

template<class scalar_t>
void magmaLuNoPivBatched(
    magma_int_t m, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue);

template<class scalar_t>
inline magma_int_t magmaGetriOptimalBlocksize(magma_int_t n);

template<class scalar_t>
void magmaGetri(
    magma_int_t n, scalar_t* dA, magma_int_t ldda, magma_int_t* ipiv, scalar_t* dwork,
    magma_int_t lwork, magma_int_t* info);

template<class scalar_t>
void magmaGetriBatched(
    magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, scalar_t** dinvA_array, magma_int_t lddia,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue);

template<class scalar_t>
void magmaCholeskySolve(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, scalar_t* dA, magma_int_t ldda,
    scalar_t* dB, magma_int_t lddb, magma_int_t* info);

template<class scalar_t>
void magmaCholeskySolveBatched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, scalar_t** dA_array, magma_int_t ldda,
    scalar_t** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue);

template<class scalar_t>
void magmaCholesky(
    magma_uplo_t uplo, magma_int_t n, scalar_t* dA,
    magma_int_t ldda, magma_int_t* info);

template<class scalar_t>
void magmaCholeskyBatched(
    magma_uplo_t uplo, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue);

template<class scalar_t>
void magmaTriangularSolve(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    scalar_t* dA, magma_int_t ldda, scalar_t* dB, magma_int_t lddb);

template<class scalar_t>
void magmaTriangularSolveBatched(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    scalar_t** dA_array, magma_int_t ldda, scalar_t** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue);

template<class scalar_t>
inline magma_int_t magmaGeqrfOptimalBlocksize(magma_int_t m, magma_int_t n);

template<class scalar_t>
void magmaGeqrf(
    magma_int_t m, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    scalar_t* tau, scalar_t* dT, magma_int_t* info, bool is_v2);

template<class scalar_t>
void magmaOrgqr(
    magma_int_t m, magma_int_t n, magma_int_t k, scalar_t* dA,
    magma_int_t ldda, scalar_t* tau, scalar_t* dT, magma_int_t nb, magma_int_t* info);

template<class scalar_t, class value_t=scalar_t>
void magmaSymeig(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    value_t* w, scalar_t* wA, magma_int_t ldwa, scalar_t* work, magma_int_t lwork, value_t* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info);

template<class scalar_t>
void magmaEig(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n, scalar_t *A, magma_int_t lda,
    scalar_t *wr, scalar_t *wi, scalar_t *VL, magma_int_t ldvl,
    scalar_t *VR, magma_int_t ldvr, scalar_t *work, magma_int_t lwork, magma_int_t *info);

template<class scalar_t, class value_t=scalar_t>
void magmaSvd(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, scalar_t* A,
    magma_int_t lda, value_t* s, scalar_t* U, magma_int_t ldu,
    scalar_t* VT, magma_int_t ldvt, scalar_t* work, magma_int_t lwork,
    value_t* rwork,
    magma_int_t* iwork, magma_int_t* info);

template<class scalar_t>
void magmaLuSolve(
    magma_int_t n, magma_int_t nrhs, scalar_t* dA, magma_int_t ldda, magma_int_t* ipiv,
    scalar_t* dB, magma_int_t lddb, magma_int_t* info);

template<class scalar_t>
void magmaLuSolveBatched(
    magma_int_t n, magma_int_t nrhs, scalar_t** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    scalar_t** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue);

template<>
void magmaSolve<double>(
    magma_int_t n, magma_int_t nrhs, double* dA, magma_int_t ldda,
    magma_int_t* ipiv, double* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dgesv_gpu(n, nrhs, dA, ldda, ipiv, dB, lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSolve<float>(
    magma_int_t n, magma_int_t nrhs, float* dA, magma_int_t ldda,
    magma_int_t* ipiv, float* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_sgesv_gpu(n, nrhs, dA, ldda, ipiv, dB, lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSolve<c10::complex<double>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<double>* dA, magma_int_t ldda,
    magma_int_t* ipiv, c10::complex<double>* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgesv_gpu(n, nrhs,
    reinterpret_cast<magmaDoubleComplex*>(dA), ldda, ipiv,
    reinterpret_cast<magmaDoubleComplex*>(dB), lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSolve<c10::complex<float>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<float>* dA, magma_int_t ldda,
    magma_int_t* ipiv, c10::complex<float>* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgesv_gpu(n, nrhs,
    reinterpret_cast<magmaFloatComplex*>(dA), ldda, ipiv,
    reinterpret_cast<magmaFloatComplex*>(dB), lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSolveBatched<double>(
    magma_int_t n, magma_int_t nrhs, double** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, double** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, const MAGMAQueue& magma_queue) {
  magma_dgesv_batched(n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, dinfo_array, batch_count, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSolveBatched<float>(
    magma_int_t n, magma_int_t nrhs, float** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, float** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, const MAGMAQueue& magma_queue) {
  magma_sgesv_batched(n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, dinfo_array, batch_count, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSolveBatched<c10::complex<double>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<double>** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, c10::complex<double>** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, const MAGMAQueue& magma_queue) {
  magma_zgesv_batched(n, nrhs,
    reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, dipiv_array,
    reinterpret_cast<magmaDoubleComplex**>(dB_array), lddb, dinfo_array, batch_count, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSolveBatched<c10::complex<float>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<float>** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, c10::complex<float>** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, const MAGMAQueue& magma_queue) {
  magma_cgesv_batched(n, nrhs,
    reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, dipiv_array,
    reinterpret_cast<magmaFloatComplex**>(dB_array), lddb, dinfo_array, batch_count, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLu<double>(
    magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dgetrf_gpu(m, n, dA, ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLu<float>(
    magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_sgetrf_gpu(m, n, dA, ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLu<c10::complex<double>>(
    magma_int_t m, magma_int_t n, c10::complex<double>* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgetrf_gpu(m, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLu<c10::complex<float>>(
    magma_int_t m, magma_int_t n, c10::complex<float>* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgetrf_gpu(m, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuBatched<double>(
    magma_int_t m, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magma_dgetrf_batched(m, n, dA_array, ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuBatched<float>(
    magma_int_t m, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magma_sgetrf_batched(m, n, dA_array, ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuBatched<c10::complex<double>>(
    magma_int_t m, magma_int_t n, c10::complex<double>** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magma_zgetrf_batched(m, n, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuBatched<c10::complex<float>>(
    magma_int_t m, magma_int_t n, c10::complex<float>** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magma_cgetrf_batched(m, n, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<double>(
    magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dgetrf_nopiv_gpu(m, n, dA, ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<float>(
    magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_sgetrf_nopiv_gpu(m, n, dA, ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<c10::complex<double>>(
    magma_int_t m, magma_int_t n, c10::complex<double>* dA, magma_int_t ldda,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgetrf_nopiv_gpu(m, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<c10::complex<float>>(
    magma_int_t m, magma_int_t n, c10::complex<float>* dA, magma_int_t ldda,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgetrf_nopiv_gpu(m, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<double>(
    magma_int_t m, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_dgetrf_nopiv_batched(m, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<float>(
    magma_int_t m, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_sgetrf_nopiv_batched(m, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<c10::complex<double>>(
    magma_int_t m, magma_int_t n, c10::complex<double>** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_zgetrf_nopiv_batched(m, n, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<c10::complex<float>>(
    magma_int_t m, magma_int_t n, c10::complex<float>** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_cgetrf_nopiv_batched(m, n, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
inline magma_int_t magmaGetriOptimalBlocksize<double>(magma_int_t n) {
  return magma_get_dgetri_nb(n);
}

template<>
inline magma_int_t magmaGetriOptimalBlocksize<float>(magma_int_t n) {
  return magma_get_sgetri_nb(n);
}

template <>
inline magma_int_t magmaGetriOptimalBlocksize<c10::complex<double>>(
    magma_int_t n) {
  return magma_get_zgetri_nb(n);
}

template <>
inline magma_int_t magmaGetriOptimalBlocksize<c10::complex<float>>(
    magma_int_t n) {
  return magma_get_cgetri_nb(n);
}

template<>
void magmaGetri<double>(
    magma_int_t n, double* dA, magma_int_t ldda, magma_int_t* ipiv, double* dwork,
    magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dgetri_gpu(n, dA, ldda, ipiv, dwork, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaGetri<float>(
    magma_int_t n, float* dA, magma_int_t ldda, magma_int_t* ipiv, float* dwork,
    magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_sgetri_gpu(n, dA, ldda, ipiv, dwork, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaGetri<c10::complex<double>>(
    magma_int_t n,
    c10::complex<double>* dA,
    magma_int_t ldda,
    magma_int_t* ipiv,
    c10::complex<double>* dwork,
    magma_int_t lwork,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgetri_gpu(
      n,
      reinterpret_cast<magmaDoubleComplex*>(dA),
      ldda,
      ipiv,
      reinterpret_cast<magmaDoubleComplex*>(dwork),
      lwork,
      info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaGetri<c10::complex<float>>(
    magma_int_t n,
    c10::complex<float>* dA,
    magma_int_t ldda,
    magma_int_t* ipiv,
    c10::complex<float>* dwork,
    magma_int_t lwork,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgetri_gpu(
      n,
      reinterpret_cast<magmaFloatComplex*>(dA),
      ldda,
      ipiv,
      reinterpret_cast<magmaFloatComplex*>(dwork),
      lwork,
      info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaGetriBatched<double>(
    magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, double** dinvA_array, magma_int_t lddia,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_dgetri_outofplace_batched(n, dA_array, ldda, ipiv_array, dinvA_array, lddia, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaGetriBatched<float>(
    magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, float** dinvA_array, magma_int_t lddia,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_sgetri_outofplace_batched(n, dA_array, ldda, ipiv_array, dinvA_array, lddia, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaGetriBatched<c10::complex<double>>(
    magma_int_t n,
    c10::complex<double>** dA_array,
    magma_int_t ldda,
    magma_int_t** ipiv_array,
    c10::complex<double>** dinvA_array,
    magma_int_t lddia,
    magma_int_t* info_array,
    magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magma_zgetri_outofplace_batched(
      n,
      reinterpret_cast<magmaDoubleComplex**>(dA_array),
      ldda,
      ipiv_array,
      reinterpret_cast<magmaDoubleComplex**>(dinvA_array),
      lddia,
      info_array,
      batchsize,
      magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaGetriBatched<c10::complex<float>>(
    magma_int_t n,
    c10::complex<float>** dA_array,
    magma_int_t ldda,
    magma_int_t** ipiv_array,
    c10::complex<float>** dinvA_array,
    magma_int_t lddia,
    magma_int_t* info_array,
    magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magma_cgetri_outofplace_batched(
      n,
      reinterpret_cast<magmaFloatComplex**>(dA_array),
      ldda,
      ipiv_array,
      reinterpret_cast<magmaFloatComplex**>(dinvA_array),
      lddia,
      info_array,
      batchsize,
      magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<double>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, double* dA, magma_int_t ldda,
    double* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dpotrs_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<float>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, float* dA, magma_int_t ldda,
    float* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_spotrs_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<c10::complex<double>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<double>* dA, magma_int_t ldda,
    c10::complex<double>* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zpotrs_gpu(uplo, n, nrhs,
    reinterpret_cast<magmaDoubleComplex*>(dA), ldda,
    reinterpret_cast<magmaDoubleComplex*>(dB), lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<float>* dA, magma_int_t ldda,
    c10::complex<float>* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cpotrs_gpu(uplo, n, nrhs,
    reinterpret_cast<magmaFloatComplex*>(dA), ldda,
    reinterpret_cast<magmaFloatComplex*>(dB), lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolveBatched<double>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, double** dA_array, magma_int_t ldda,
    double** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  info = magma_dpotrs_batched(uplo, n, nrhs, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolveBatched<float>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, float** dA_array, magma_int_t ldda,
    float** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  info = magma_spotrs_batched(uplo, n, nrhs, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolveBatched<c10::complex<double>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<double>** dA_array, magma_int_t ldda,
    c10::complex<double>** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  info = magma_zpotrs_batched(uplo, n, nrhs,
    reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda,
    reinterpret_cast<magmaDoubleComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolveBatched<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<float>** dA_array, magma_int_t ldda,
    c10::complex<float>** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  info = magma_cpotrs_batched(uplo, n, nrhs,
    reinterpret_cast<magmaFloatComplex**>(dA_array), ldda,
    reinterpret_cast<magmaFloatComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholesky<double>(
    magma_uplo_t uplo, magma_int_t n, double* dA,
    magma_int_t ldda, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dpotrf_gpu(uplo, n, dA, ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholesky<float>(
    magma_uplo_t uplo, magma_int_t n, float* dA,
    magma_int_t ldda, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_spotrf_gpu(uplo, n, dA, ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholesky<c10::complex<double>>(
    magma_uplo_t uplo, magma_int_t n, c10::complex<double>* dA,
    magma_int_t ldda, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zpotrf_gpu(uplo, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholesky<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, c10::complex<float>* dA,
    magma_int_t ldda, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cpotrf_gpu(uplo, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskyBatched<double>(
    magma_uplo_t uplo, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_dpotrf_batched(uplo, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskyBatched<float>(
    magma_uplo_t uplo, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_spotrf_batched(uplo, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskyBatched<c10::complex<double>>(
    magma_uplo_t uplo, magma_int_t n, c10::complex<double>** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_zpotrf_batched(uplo, n, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskyBatched<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, c10::complex<float>** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_cpotrf_batched(uplo, n, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolve<double>(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    double* dA, magma_int_t ldda, double* dB, magma_int_t lddb) {
  MagmaStreamSyncGuard guard;
  magma_dtrsm(MagmaLeft, uplo, trans, diag, m, n, 1, dA, ldda, dB, lddb);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolve<float>(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    float* dA, magma_int_t ldda, float* dB, magma_int_t lddb) {
  MagmaStreamSyncGuard guard;
  magma_strsm(MagmaLeft, uplo, trans, diag, m, n, 1, dA, ldda, dB, lddb);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolve<c10::complex<double>>(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    c10::complex<double>* dA, magma_int_t ldda, c10::complex<double>* dB, magma_int_t lddb) {
  MagmaStreamSyncGuard guard;
  magmaDoubleComplex alpha({1, 0});
  magma_ztrsm(MagmaLeft, uplo, trans, diag, m, n, alpha,
    reinterpret_cast<magmaDoubleComplex*>(dA), ldda,
    reinterpret_cast<magmaDoubleComplex*>(dB), lddb);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolve<c10::complex<float>>(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    c10::complex<float>* dA, magma_int_t ldda, c10::complex<float>* dB, magma_int_t lddb) {
  MagmaStreamSyncGuard guard;
  magmaFloatComplex alpha({1, 0});
  magma_ctrsm(MagmaLeft, uplo, trans, diag, m, n, alpha,
    reinterpret_cast<magmaFloatComplex*>(dA), ldda,
    reinterpret_cast<magmaFloatComplex*>(dB), lddb);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolveBatched<double>(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    double** dA_array, magma_int_t ldda, double** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magmablas_dtrsm_batched(MagmaLeft, uplo, trans, diag, m, n, 1, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolveBatched<float>(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    float** dA_array, magma_int_t ldda, float** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magmablas_strsm_batched(MagmaLeft, uplo, trans, diag, m, n, 1, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolveBatched<c10::complex<double>>(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    c10::complex<double>** dA_array, magma_int_t ldda, c10::complex<double>** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magmaDoubleComplex alpha({1, 0});
  magmablas_ztrsm_batched(MagmaLeft, uplo, trans, diag, m, n, alpha,
    reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda,
    reinterpret_cast<magmaDoubleComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolveBatched<c10::complex<float>>(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    c10::complex<float>** dA_array, magma_int_t ldda, c10::complex<float>** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magmaFloatComplex alpha({1, 0});
  magmablas_ctrsm_batched(MagmaLeft, uplo, trans, diag, m, n, alpha,
    reinterpret_cast<magmaFloatComplex**>(dA_array), ldda,
    reinterpret_cast<magmaFloatComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
inline magma_int_t magmaGeqrfOptimalBlocksize<double>(magma_int_t m, magma_int_t n) {
  return magma_get_dgeqrf_nb(m, n);
}

template<>
inline magma_int_t magmaGeqrfOptimalBlocksize<float>(magma_int_t m, magma_int_t n) {
  return magma_get_sgeqrf_nb(m, n);
}

template <>
inline magma_int_t magmaGeqrfOptimalBlocksize<c10::complex<double>>(
    magma_int_t m,
    magma_int_t n) {
  return magma_get_zgeqrf_nb(m, n);
}

template <>
inline magma_int_t magmaGeqrfOptimalBlocksize<c10::complex<float>>(
    magma_int_t m,
    magma_int_t n) {
  return magma_get_cgeqrf_nb(m, n);
}

template<>
void magmaGeqrf<double>(
    magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda,
    double* tau, double* dT, magma_int_t* info, bool is_v2) {
  MagmaStreamSyncGuard guard;
  if (!is_v2) {
    magma_dgeqrf_gpu(m, n, dA, ldda, tau, dT, info);
  } else {
    magma_dgeqrf2_gpu(m, n, dA, ldda, tau, info);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaGeqrf<float>(
    magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda,
    float* tau, float* dT, magma_int_t* info, bool is_v2) {
  MagmaStreamSyncGuard guard;
  if (!is_v2) {
    magma_sgeqrf_gpu(m, n, dA, ldda, tau, dT, info);
  } else {
    magma_sgeqrf2_gpu(m, n, dA, ldda, tau, info);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaGeqrf<c10::complex<double>>(
    magma_int_t m,
    magma_int_t n,
    c10::complex<double>* dA,
    magma_int_t ldda,
    c10::complex<double>* tau,
    c10::complex<double>* dT,
    magma_int_t* info,
    bool is_v2) {
  MagmaStreamSyncGuard guard;
  if (!is_v2) {
    magma_zgeqrf_gpu(
        m,
        n,
        reinterpret_cast<magmaDoubleComplex*>(dA),
        ldda,
        reinterpret_cast<magmaDoubleComplex*>(tau),
        reinterpret_cast<magmaDoubleComplex*>(dT),
        info);
  } else {
    magma_zgeqrf2_gpu(
        m,
        n,
        reinterpret_cast<magmaDoubleComplex*>(dA),
        ldda,
        reinterpret_cast<magmaDoubleComplex*>(tau),
        info);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaGeqrf<c10::complex<float>>(
    magma_int_t m,
    magma_int_t n,
    c10::complex<float>* dA,
    magma_int_t ldda,
    c10::complex<float>* tau,
    c10::complex<float>* dT,
    magma_int_t* info,
    bool is_v2) {
  MagmaStreamSyncGuard guard;
  if (!is_v2) {
    magma_cgeqrf_gpu(
        m,
        n,
        reinterpret_cast<magmaFloatComplex*>(dA),
        ldda,
        reinterpret_cast<magmaFloatComplex*>(tau),
        reinterpret_cast<magmaFloatComplex*>(dT),
        info);
  } else {
    magma_cgeqrf2_gpu(
        m,
        n,
        reinterpret_cast<magmaFloatComplex*>(dA),
        ldda,
        reinterpret_cast<magmaFloatComplex*>(tau),
        info);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaOrgqr<double>(
    magma_int_t m, magma_int_t n, magma_int_t k, double* dA, magma_int_t ldda,
    double* tau, double* dT, magma_int_t nb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dorgqr_gpu(m, n, k, dA, ldda, tau, dT, nb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaOrgqr<float>(
    magma_int_t m, magma_int_t n, magma_int_t k, float* dA, magma_int_t ldda,
    float* tau, float* dT, magma_int_t nb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_sorgqr_gpu(m, n, k, dA, ldda, tau, dT, nb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaOrgqr<c10::complex<double>>(
    magma_int_t m,
    magma_int_t n,
    magma_int_t k,
    c10::complex<double>* dA,
    magma_int_t ldda,
    c10::complex<double>* tau,
    c10::complex<double>* dT,
    magma_int_t nb,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zungqr_gpu(
      m,
      n,
      k,
      reinterpret_cast<magmaDoubleComplex*>(dA),
      ldda,
      reinterpret_cast<magmaDoubleComplex*>(tau),
      reinterpret_cast<magmaDoubleComplex*>(dT),
      nb,
      info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaOrgqr<c10::complex<float>>(
    magma_int_t m,
    magma_int_t n,
    magma_int_t k,
    c10::complex<float>* dA,
    magma_int_t ldda,
    c10::complex<float>* tau,
    c10::complex<float>* dT,
    magma_int_t nb,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cungqr_gpu(
    m,
    n,
    k,
    reinterpret_cast<magmaFloatComplex*>(dA),
    ldda,
    reinterpret_cast<magmaFloatComplex*>(tau),
    reinterpret_cast<magmaFloatComplex*>(dT),
    nb,
    info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSymeig<double>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, double* dA, magma_int_t ldda,
    double* w, double* wA, magma_int_t ldwa, double* work, magma_int_t lwork, double* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  (void)rwork;  // unused
  (void)lrwork;  // unused
  MagmaStreamSyncGuard guard;
  magma_dsyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork, liwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSymeig<float>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, float* dA, magma_int_t ldda,
    float* w, float* wA, magma_int_t ldwa, float* work, magma_int_t lwork, float* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  (void)rwork;  // unused
  (void)lrwork;  // unused
  MagmaStreamSyncGuard guard;
  magma_ssyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork, liwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSymeig<c10::complex<double>, double>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, c10::complex<double>* dA, magma_int_t ldda,
    double* w, c10::complex<double>* wA, magma_int_t ldwa, c10::complex<double>* work, magma_int_t lwork, double* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zheevd_gpu(
      jobz, uplo, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, w, reinterpret_cast<magmaDoubleComplex*>(wA),
      ldwa, reinterpret_cast<magmaDoubleComplex*>(work), lwork, rwork, lrwork, iwork, liwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSymeig<c10::complex<float>, float>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, c10::complex<float>* dA, magma_int_t ldda,
    float* w, c10::complex<float>* wA, magma_int_t ldwa, c10::complex<float>* work, magma_int_t lwork, float* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cheevd_gpu(
      jobz, uplo, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, w, reinterpret_cast<magmaFloatComplex*>(wA),
      ldwa, reinterpret_cast<magmaFloatComplex*>(work), lwork, rwork, lrwork, iwork, liwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}
    
template<>
void magmaEig<double>(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n, double *A, magma_int_t lda,
    double *wr, double *wi, double *VL, magma_int_t ldvl,
    double *VR, magma_int_t ldvr, double *work, magma_int_t lwork, magma_int_t *info) {
  MagmaStreamSyncGuard guard;
  magma_dgeev(jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr, work, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaEig<float>(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n, float *A, magma_int_t lda,
    float *wr, float *wi, float *VL, magma_int_t ldvl,
    float *VR, magma_int_t ldvr, float *work, magma_int_t lwork, magma_int_t *info) {
  MagmaStreamSyncGuard guard;
  magma_sgeev(jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr, work, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSvd<double>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, double* A,
    magma_int_t lda, double* s, double* U, magma_int_t ldu,
    double* VT, magma_int_t ldvt, double* work, magma_int_t lwork,
    double *rwork, magma_int_t* iwork, magma_int_t* info) {
  (void)rwork; // unused
  MagmaStreamSyncGuard guard;
  magma_dgesdd(jobz, m, n, A, lda, s, U, ldu, VT, ldvt, work, lwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSvd<float>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, float* A,
    magma_int_t lda, float* s, float* U, magma_int_t ldu,
    float* VT, magma_int_t ldvt, float* work, magma_int_t lwork,
    float* rwork, magma_int_t* iwork, magma_int_t* info) {
  (void)rwork; // unused
  MagmaStreamSyncGuard guard;
  magma_sgesdd(jobz, m, n, A, lda, s, U, ldu, VT, ldvt, work, lwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSvd<c10::complex<float>, float>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, c10::complex<float>* A,
    magma_int_t lda, float* s, c10::complex<float>* U, magma_int_t ldu,
    c10::complex<float>* VT, magma_int_t ldvt, c10::complex<float>* work, magma_int_t lwork,
    float *rwork, magma_int_t* iwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgesdd(jobz, m, n, reinterpret_cast<magmaFloatComplex*>(A), lda, s,
                reinterpret_cast<magmaFloatComplex*>(U), ldu,
                reinterpret_cast<magmaFloatComplex*>(VT), ldvt,
                reinterpret_cast<magmaFloatComplex*>(work), lwork,
                rwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSvd<c10::complex<double>, double>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, c10::complex<double>* A,
    magma_int_t lda, double* s, c10::complex<double>* U, magma_int_t ldu,
    c10::complex<double>* VT, magma_int_t ldvt, c10::complex<double>* work, magma_int_t lwork,
    double *rwork, magma_int_t* iwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgesdd(jobz, m, n, reinterpret_cast<magmaDoubleComplex*>(A), lda, s,
                reinterpret_cast<magmaDoubleComplex*>(U), ldu,
                reinterpret_cast<magmaDoubleComplex*>(VT), ldvt,
                reinterpret_cast<magmaDoubleComplex*>(work), lwork,
                rwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<double>(
    magma_int_t n, magma_int_t nrhs, double* dA, magma_int_t ldda, magma_int_t* ipiv,
    double* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dgetrs_gpu(MagmaNoTrans, n, nrhs, dA, ldda, ipiv, dB, lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<float>(
    magma_int_t n, magma_int_t nrhs, float* dA, magma_int_t ldda, magma_int_t* ipiv,
    float* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_sgetrs_gpu(MagmaNoTrans, n, nrhs, dA, ldda, ipiv, dB, lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<c10::complex<double>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<double>* dA, magma_int_t ldda, magma_int_t* ipiv,
    c10::complex<double>* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgetrs_gpu(MagmaNoTrans, n, nrhs, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, ipiv, reinterpret_cast<magmaDoubleComplex*>(dB), lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<c10::complex<float>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<float>* dA, magma_int_t ldda, magma_int_t* ipiv,
    c10::complex<float>* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgetrs_gpu(MagmaNoTrans, n, nrhs, reinterpret_cast<magmaFloatComplex*>(dA), ldda, ipiv, reinterpret_cast<magmaFloatComplex*>(dB), lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolveBatched<double>(
    magma_int_t n, magma_int_t nrhs, double** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    double** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  info = magma_dgetrs_batched(MagmaNoTrans, n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolveBatched<float>(
    magma_int_t n, magma_int_t nrhs, float** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    float** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue) {
 info = magma_sgetrs_batched(MagmaNoTrans, n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, batchsize, magma_queue.get_queue());
 AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolveBatched<c10::complex<double>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<double>** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    c10::complex<double>** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  info = magma_zgetrs_batched(MagmaNoTrans, n, nrhs, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, dipiv_array, reinterpret_cast<magmaDoubleComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolveBatched<c10::complex<float>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<float>** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    c10::complex<float>** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue) {
 info = magma_cgetrs_batched(MagmaNoTrans, n, nrhs, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, dipiv_array, reinterpret_cast<magmaFloatComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
 AT_CUDA_CHECK(cudaGetLastError());
}
#endif

#define ALLOCATE_ARRAY(name, type, size) \
  auto storage_##name = pin_memory<type>(size); \
  name = static_cast<type*>(storage_##name.data());

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_solve(Tensor& b, Tensor& A, std::vector<int64_t>& infos) {
#ifndef USE_MAGMA
AT_ERROR("solve: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  auto A_data = A.data_ptr<scalar_t>();
  auto b_data = b.data_ptr<scalar_t>();
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");
  magma_int_t nrhs = magma_int_cast(b.size(-1), "b.size(-1)");
  magma_int_t lda = std::max(magma_int_t{1}, n);

  if (b.dim() == 2) {
    auto ipiv = at::empty({n}, at::kInt);
    magma_int_t info = 0;
    magmaSolve<scalar_t>(n, nrhs, A_data, lda, ipiv.data_ptr<magma_int_t>(),
                        b_data, lda, &info);
    infos[0] = info;
  } else {
    auto A_mat_stride = matrixStride(A);
    auto b_mat_stride = matrixStride(b);
    magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");

    magma_int_t* info_array;
    magma_int_t* ipiv_data;
    magma_int_t** ipiv_array;
    scalar_t** A_array;
    scalar_t** b_array;

    ALLOCATE_ARRAY(info_array, magma_int_t, batch_size);
    ALLOCATE_ARRAY(ipiv_data, magma_int_t, batch_size * n);
    ALLOCATE_ARRAY(ipiv_array, magma_int_t*, batch_size);
    ALLOCATE_ARRAY(A_array, scalar_t*, batch_size);
    ALLOCATE_ARRAY(b_array, scalar_t*, batch_size);

    // Set up the created arrays
    for (int64_t i = 0; i < batch_size; i++) {
      A_array[i] = &A_data[i * A_mat_stride];
      b_array[i] = &b_data[i * b_mat_stride];
      ipiv_array[i] = &ipiv_data[i * n];
    }

    MAGMAQueue magma_queue(b.get_device());

    constexpr int64_t batch_limit = 65535;
    // Compute as many batches of 65535 possible
    // The number of "mini"-batches are floor(batch_size / batch_limit)
    // and these cover floor(batch_size / batch_limit) * batch_limit matrix solves
    int64_t mini_batches = batch_size / batch_limit, mini_idx;
    for (mini_idx = 0; mini_idx < mini_batches * batch_limit; mini_idx += batch_limit) {
      scalar_t** A_array_cur = &A_array[mini_idx];
      scalar_t** b_array_cur = &b_array[mini_idx];
      magma_int_t** ipiv_array_cur = &ipiv_array[mini_idx];
      magma_int_t* info_array_cur = &info_array[mini_idx];

      magmaSolveBatched<scalar_t>(
          n, nrhs, A_array_cur, lda, ipiv_array_cur, b_array_cur, lda,
          info_array_cur, batch_limit, magma_queue);
    }

    // Compute whatever is left = batch_size - floor(batch_size / batch_limit) * batch_limit
    // which concisely is equal to batch_size % batch_limit
    if (batch_size % batch_limit != 0) {
      magmaSolveBatched<scalar_t>(
          n, nrhs, &A_array[mini_idx], lda, &ipiv_array[mini_idx], &b_array[mini_idx], lda,
          &info_array[mini_idx], batch_size % batch_limit, magma_queue);
    }

    for (int64_t i = 0; i < batch_size; i++) {
      infos[i] = info_array[i];
    }
  }
#endif
}

std::tuple<Tensor, Tensor> _solve_helper_cuda(const Tensor& self, const Tensor& A) {
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  std::vector<int64_t> infos(batchCount(self), 0);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "solve_cuda", [&]{
    apply_solve<scalar_t>(self_working_copy, A_working_copy, infos);
  });
  if (self.dim() > 2) {
    batchCheckErrors(infos, "solve_cuda");
  } else {
    singleCheckErrors(infos[0], "solve_cuda");
  }
  return std::tuple<Tensor, Tensor>(self_working_copy, A_working_copy);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ inverse ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_batched_inverse(Tensor& self, Tensor& self_inv, std::vector<int64_t>& infos) {
#ifndef USE_MAGMA
AT_ERROR("inverse: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  auto self_data = self.data_ptr<scalar_t>();
  auto self_mat_stride = matrixStride(self);
  auto self_inv_data = self_inv.data_ptr<scalar_t>();
  auto self_inv_mat_stride = matrixStride(self_inv);

  magma_int_t batch_size = magma_int_cast(batchCount(self), "batchCount");
  magma_int_t n = magma_int_cast(self.size(-2), "self.size(-2)");

  magma_int_t* info_array;
  magma_int_t* ipiv_data;
  magma_int_t** ipiv_array;
  scalar_t** self_array;
  scalar_t** self_inv_array;

  ALLOCATE_ARRAY(info_array, magma_int_t, batch_size);
  ALLOCATE_ARRAY(ipiv_data, magma_int_t, batch_size * n);
  ALLOCATE_ARRAY(ipiv_array, magma_int_t*, batch_size);
  ALLOCATE_ARRAY(self_array, scalar_t*, batch_size);
  ALLOCATE_ARRAY(self_inv_array, scalar_t*, batch_size);

  // Set up the created arrays
  for (int64_t i = 0; i < batch_size; i++) {
    self_array[i] = &self_data[i * self_mat_stride];
    self_inv_array[i] = &self_inv_data[i * self_inv_mat_stride];
    ipiv_array[i] = &ipiv_data[i * n];
  }

  MAGMAQueue magma_queue(self.get_device());
  magmaLuBatched<scalar_t>(
    n, n, self_array, n, ipiv_array, info_array,
    batch_size, magma_queue);

  constexpr int64_t batch_limit = 65535;
  // Compute as many batches of 65535 possible
  // The number of "mini"-batches are floor(batch_size / batch_limit)
  // and these cover floor(batch_size / batch_limit) * batch_limit matrix solves
  int64_t mini_batches = batch_size / batch_limit, mini_idx;
  for (mini_idx = 0; mini_idx < mini_batches * batch_limit; mini_idx += batch_limit) {
    scalar_t** self_array_cur = &self_array[mini_idx];
    scalar_t** self_inv_array_cur = &self_inv_array[mini_idx];
    magma_int_t** ipiv_array_cur = &ipiv_array[mini_idx];
    magma_int_t* info_array_cur = &info_array[mini_idx];

    magmaGetriBatched<scalar_t>(
      n, self_array_cur, n, ipiv_array_cur, self_inv_array_cur,
      n, info_array_cur, batch_limit, magma_queue);
  }

  // Compute whatever is left = batch_size - floor(batch_size / batch_limit) * batch_limit
  // which concisely is equal to batch_size % batch_limit
  if (batch_size % batch_limit != 0) {
    magmaGetriBatched<scalar_t>(
      n, &self_array[mini_idx], n, &ipiv_array[mini_idx], &self_inv_array[mini_idx],
      n, &info_array[mini_idx], batch_size % batch_limit, magma_queue);
  }

  for (int64_t i = 0; i < batch_size; i++) {
    infos[i] = info_array[i];
  }
#endif
}

template <typename scalar_t>
static void apply_single_inverse(Tensor& self, int64_t& info) {
#ifndef USE_MAGMA
AT_ERROR("inverse: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  auto self_data = self.data_ptr<scalar_t>();
  magma_int_t n = magma_int_cast(self.size(-2), "self.size(-2)");
  magma_int_t lwork = n * magmaGetriOptimalBlocksize<scalar_t>(n);
  magma_int_t info_tmp = 0;

  Tensor ipiv = at::empty({n}, at::kInt);
  Tensor dwork = at::empty({lwork}, self.options());
  magmaLu<scalar_t>(n, n, self_data, n, ipiv.data_ptr<magma_int_t>(), &info_tmp);
  if (info_tmp != 0) {
    info = info_tmp;
    return;
  }
  magmaGetri<scalar_t>(
    n, self_data, n, ipiv.data_ptr<magma_int_t>(), dwork.data_ptr<scalar_t>(), lwork, &info_tmp);
  info = info_tmp;
#endif
}

Tensor _inverse_helper_cuda_legacy(const Tensor& self) {
  auto self_inv_working_copy = cloneBatchedColumnMajor(self);
  if (self.dim() > 2) {
    std::vector<int64_t> infos(batchCount(self), 0);
    auto self_working_copy = cloneBatchedColumnMajor(self);
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "inverse_cuda", [&]{
      apply_batched_inverse<scalar_t>(
        self_working_copy, self_inv_working_copy, infos);
    });
    batchCheckErrors(infos, "inverse_cuda");
  } else {
    int64_t info = 0;
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "inverse_cuda", [&]{
      apply_single_inverse<scalar_t>(self_inv_working_copy, info);
    });
    singleCheckErrors(info, "inverse_cuda");
  }
  return self_inv_working_copy;
}

Tensor _inverse_helper_cuda(const Tensor& self) {
#ifdef USE_CUSOLVER
  if ((self.dim() == 2) || (/* self.dim() > 2 && */ batchCount(self) <= 2) || !use_magma_) {
    return _inverse_helper_cuda_lib(self);    // cusolver or cublas
  } else {
    return _inverse_helper_cuda_legacy(self); // magma-cuda
  }
#else
  return _inverse_helper_cuda_legacy(self); // magma-cuda
#endif
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_cholesky_solve(Tensor& b, Tensor& A, bool upper, int64_t& info) {
#ifndef USE_MAGMA
AT_ERROR("cholesky_solve: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;

  auto A_data = A.data_ptr<scalar_t>();
  auto b_data = b.data_ptr<scalar_t>();
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");
  magma_int_t nrhs = magma_int_cast(b.size(-1), "b.size(-1)");

  int info_tmp = 0;
  if (b.dim() == 2) {
    magmaCholeskySolve<scalar_t>(uplo, n, nrhs, A_data, n,
                                 b_data, n, &info_tmp);
    info = info_tmp;
  } else {
    auto A_mat_stride = matrixStride(A);
    auto b_mat_stride = matrixStride(b);
    magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");

    scalar_t** A_array;
    scalar_t** b_array;

    ALLOCATE_ARRAY(A_array, scalar_t*, batch_size);
    ALLOCATE_ARRAY(b_array, scalar_t*, batch_size);

    // Set up the created arrays
    for (int64_t i = 0; i < batch_size; i++) {
      A_array[i] = &A_data[i * A_mat_stride];
      b_array[i] = &b_data[i * b_mat_stride];
    }

    MAGMAQueue magma_queue(b.get_device());

    constexpr int64_t batch_limit = 65535;
    // Compute as many batches of 65535 possible
    // The number of "mini"-batches are floor(batch_size / batch_limit)
    // and these cover floor(batch_size / batch_limit) * batch_limit matrix solves
    int64_t mini_batches = batch_size / batch_limit, mini_idx;
    for (mini_idx = 0; mini_idx < mini_batches * batch_limit; mini_idx += batch_limit) {
      scalar_t** A_array_cur = &A_array[mini_idx];
      scalar_t** b_array_cur = &b_array[mini_idx];

      magmaCholeskySolveBatched<scalar_t>(
          uplo, n, nrhs, A_array_cur, n, b_array_cur, n,
          info_tmp, batch_limit, magma_queue);

      if (info_tmp != 0) {
        break;
      }
    }

    // Compute whatever is left = batch_size - floor(batch_size / batch_limit) * batch_limit
    // which concisely is equal to batch_size % batch_limit
    if (batch_size % batch_limit != 0 && info_tmp == 0) {
      magmaCholeskySolveBatched<scalar_t>(
          uplo, n, nrhs, &A_array[mini_idx], n, &b_array[mini_idx], n,
          info_tmp, batch_size % batch_limit, magma_queue);
    }

    info = info_tmp;
  }
#endif
}

Tensor _cholesky_solve_helper_cuda(const Tensor& self, const Tensor& A, bool upper) {
  int64_t info = 0;
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "cholesky_solve_cuda", [&]{
    apply_cholesky_solve<scalar_t>(self_working_copy, A_working_copy, upper, info);
  });
  TORCH_CHECK(info == 0, "MAGMA cholesky_solve : invalid argument: ", -info);
  return self_working_copy;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_cholesky(Tensor& self, bool upper, std::vector<int64_t>& infos) {
#ifndef USE_MAGMA
AT_ERROR("cholesky: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;

  auto self_data = self.data_ptr<scalar_t>();
  magma_int_t n = magma_int_cast(self.size(-2), "self.size(-2)");
  auto lda = std::max<magma_int_t>(1, n);

  if (self.dim() == 2) {
    magma_int_t info = 0;
    magmaCholesky<scalar_t>(uplo, n, self_data, lda, &info);
    infos[0] = info;
  } else {
    auto self_mat_stride = matrixStride(self);
    magma_int_t batch_size = magma_int_cast(batchCount(self), "batchCount");

    magma_int_t* info_array;
    scalar_t** self_array;

    ALLOCATE_ARRAY(info_array, magma_int_t, batch_size);
    ALLOCATE_ARRAY(self_array, scalar_t*, batch_size);

    // Set up the created arrays
    for (int64_t i = 0; i < batch_size; i++) {
      self_array[i] = &self_data[i * self_mat_stride];
    }

    MAGMAQueue magma_queue(self.get_device());

    int64_t batch_limit = self.is_complex() ? 65535 : 262140;
    // Compute as many batches of 262140 possible
    // 262140 is the size of the largest batch of matrices that can be run with
    // violating maximum kernel configuration
    // For complex input the batch limit is 65535 (determined experimentally, see https://github.com/pytorch/pytorch/pull/47047#discussion_r516086923 for more information)
    // The number of "mini"-batches are floor(batch_size / batch_limit)
    // and these cover floor(batch_size / batch_limit) * batch_limit cholesky calls
    int64_t mini_batches = batch_size / batch_limit, mini_idx;
    for (mini_idx = 0; mini_idx < mini_batches * batch_limit; mini_idx += batch_limit) {
      scalar_t** self_array_cur = &self_array[mini_idx];
      magma_int_t* info_array_cur = &info_array[mini_idx];

      magmaCholeskyBatched<scalar_t>(
        uplo, n, self_array_cur, lda, info_array_cur, batch_limit, magma_queue);
    }

    // Compute whatever is left = batch_size - floor(batch_size / batch_limit) * batch_limit
    // which concisely is equal to batch_size % batch_limit
    if (batch_size % batch_limit != 0) {
      magmaCholeskyBatched<scalar_t>(
        uplo, n, &self_array[mini_idx], lda, &info_array[mini_idx], batch_size % batch_limit, magma_queue);
    }

    for (int64_t i = 0; i < batch_size; i++) {
      infos[i] = info_array[i];
    }
  }
#endif
}

Tensor _cholesky_helper_cuda(const Tensor& self, bool upper) {
  std::vector<int64_t> infos(batchCount(self), 0);
  Tensor self_working_copy;
  if (upper) {
    self_working_copy = cloneBatchedColumnMajor(self.transpose(-1, -2));
  } else {
    self_working_copy = cloneBatchedColumnMajor(self);
  }

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "cholesky_cuda", [&]{
    apply_cholesky<scalar_t>(self_working_copy, false, infos);
  });
  if (self.dim() > 2) {
    batchCheckErrors(infos, "cholesky_cuda");
  } else {
    singleCheckErrors(infos[0], "cholesky_cuda");
  }
  if (upper) {
    return self_working_copy.transpose(-1, -2);
  } else {
    return self_working_copy;
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_lu(Tensor& self, Tensor& pivots, Tensor& infos, bool get_pivots) {
#ifndef USE_MAGMA
AT_ERROR("lu: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  auto self_data = self.data_ptr<scalar_t>();
  magma_int_t m = magma_int_cast(self.size(-2), "m");
  magma_int_t n = magma_int_cast(self.size(-1), "n");
  magma_int_t k = std::min(m, n);

  if (self.dim() == 2) {
    // If `pivots` is defined, then we have to compute them.
    // magmaLu and magmaLuNoPiv use a hybrid CPU-GPU algorithm to compute
    // the partially-pivoted LU decomposition with / without pivots.
    // The driver routines magma_(d/s)getrf_(nopiv_)gpu accepts a tensor on the CPU for pivots.
    // The data is later copied back to the appropriate output tensor.
    Tensor info_tmp = at::zeros({}, at::kInt);
    if (get_pivots) {
      Tensor piv_tmp = at::empty({k}, at::kInt);
      magmaLu<scalar_t>(
        m, n, self_data, m, piv_tmp.data_ptr<magma_int_t>(), info_tmp.data_ptr<magma_int_t>());
      pivots.copy_(piv_tmp);
    } else {
      magmaLuNoPiv<scalar_t>(m, n, self_data, m, info_tmp.data_ptr<magma_int_t>());
    }
    infos.copy_(info_tmp);
  } else {
    auto self_matrix_stride = matrixStride(self);
    magma_int_t batch_size = magma_int_cast(batchCount(self), "batchCount");

    scalar_t** self_array;
    ALLOCATE_ARRAY(self_array, scalar_t*, batch_size);

    // Set up the created arrays
    for (int64_t i = 0; i < batch_size; i++) {
      self_array[i] = &self_data[i * self_matrix_stride];
    }

    MAGMAQueue magma_queue(self.get_device());

    // Same comment as in the case of single matrix above.
    if (get_pivots) {
      auto pivots_data = pivots.data_ptr<magma_int_t>();
      auto pivots_matrix_stride = pivots.size(-1);
      magma_int_t** pivots_array;
      ALLOCATE_ARRAY(pivots_array, magma_int_t*, batch_size);
      for (int64_t i = 0; i < batch_size; i++) {
        pivots_array[i] = &pivots_data[i * pivots_matrix_stride];
      }
      magmaLuBatched<scalar_t>(
        m, n, self_array, m, pivots_array,
        infos.data_ptr<magma_int_t>(), batch_size, magma_queue);
    } else {
      magmaLuNoPivBatched<scalar_t>(
        m, n, self_array, m, infos.data_ptr<magma_int_t>(),
        batch_size, magma_queue);
    }
  }
#endif
}

std::tuple<Tensor, Tensor, Tensor> _lu_with_info_cuda(const Tensor& self, bool pivot, bool check_errors) {
  TORCH_CHECK(self.dim() >= 2,
           "expected tensor with 2 or more dimensions, got size: ", self.sizes(),
           " instead");
  auto m = self.size(-2);
  auto n = self.size(-1);
  auto k = std::min(m, n);
  auto req_size = self.sizes().vec();
  req_size.pop_back();
  req_size.back() = k;
  Tensor pivots_tensor = at::arange(1, k + 1, self.options().dtype(at::kInt)).expand(req_size).contiguous();
  req_size.pop_back();
  auto infos_tensor = at::zeros(req_size, self.options().dtype(at::kInt));

  Tensor self_working_copy;
  if (self.numel() == 0) {
    self_working_copy = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  } else {
    self_working_copy = cloneBatchedColumnMajor(self);
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "lu_cuda", [&]{
        apply_lu<scalar_t>(self_working_copy, pivots_tensor, infos_tensor, pivot);
    });
  }
  if (check_errors) {
    if (self.dim() == 2) {
      singleCheckErrors(infos_tensor.item<int64_t>(), "lu", /*allow_singular=*/true);
    } else {
      batchCheckErrors(infos_tensor, "lu", /*allow_singular=*/true);
    }
  }
  return std::make_tuple(self_working_copy, pivots_tensor, infos_tensor);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangular_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_triangular_solve(Tensor& b, Tensor& A, bool upper, bool transpose, bool unitriangular) {
#ifndef USE_MAGMA
AT_ERROR("triangular_solve: MAGMA library not found in "
         "compilation. Please rebuild with MAGMA.");
#else
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;
  magma_trans_t trans = transpose ? MagmaTrans : MagmaNoTrans;
  magma_diag_t diag = unitriangular ? MagmaUnit : MagmaNonUnit;

  auto A_data = A.data_ptr<scalar_t>();
  auto b_data = b.data_ptr<scalar_t>();
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");
  magma_int_t nrhs = magma_int_cast(b.size(-1), "b.size(-1)");
  magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");

  // batch_size == 1 implies that:
  // 1. the RHS and LHS tensors have 2 dimensions, or
  // 2. the RHS and LHS tensors have more than 2 dimensions but all batch dimensions are 1
  if (batch_size == 1) {
    magmaTriangularSolve<scalar_t>(uplo, trans, diag, n, nrhs, A_data, n, b_data, n);
  } else {
    auto A_mat_stride = matrixStride(A);
    auto b_mat_stride = matrixStride(b);

    scalar_t** A_array;
    scalar_t** b_array;

    ALLOCATE_ARRAY(A_array, scalar_t*, batch_size);
    ALLOCATE_ARRAY(b_array, scalar_t*, batch_size);

    // Set up the created arrays
    for (int64_t i = 0; i < batch_size; i++) {
      A_array[i] = &A_data[i * A_mat_stride];
      b_array[i] = &b_data[i * b_mat_stride];
    }

    MAGMAQueue magma_queue(b.get_device());

    constexpr int64_t batch_limit = 65535;
    // Compute as many batches of 65535 possible
    // The number of "mini"-batches are floor(batch_size / batch_limit)
    // and these cover floor(batch_size / batch_limit) * batch_limit matrix solves
    int64_t mini_batches = batch_size / batch_limit, mini_idx;
    for (mini_idx = 0; mini_idx < mini_batches * batch_limit; mini_idx += batch_limit) {
      scalar_t** A_array_cur = &A_array[mini_idx];
      scalar_t** b_array_cur = &b_array[mini_idx];

      magmaTriangularSolveBatched<scalar_t>(
          uplo, trans, diag, n, nrhs, A_array_cur,
          n, b_array_cur, n, batch_limit, magma_queue);
    }

    // Compute whatever is left = batch_size - floor(batch_size / batch_limit) * batch_limit
    // which concisely is equal to batch_size % batch_limit
    if (batch_size % batch_limit != 0) {
      magmaTriangularSolveBatched<scalar_t>(
          uplo, trans, diag, n, nrhs, &A_array[mini_idx],
          n, &b_array[mini_idx], n, batch_size % batch_limit, magma_queue);
    }
  }
#endif
}

std::tuple<Tensor, Tensor> _triangular_solve_helper_cuda(const Tensor& self, const Tensor& A,
                                                         bool upper, bool transpose, bool unitriangular) {
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "triangular_solve_cuda", [&]{
    apply_triangular_solve<scalar_t>(self_working_copy, A_working_copy, upper, transpose, unitriangular);
  });
  return std::tuple<Tensor, Tensor>(self_working_copy, A_working_copy);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ qr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_qr(Tensor& Q, Tensor& R, int64_t n_columns, std::vector<int64_t>& infos) {
#ifndef USE_MAGMA
AT_ERROR("qr: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  auto q_data = Q.data_ptr<scalar_t>();
  auto r_data = R.data_ptr<scalar_t>();
  auto q_matrix_stride = matrixStride(Q);
  auto r_matrix_stride = matrixStride(R);

  magma_int_t m = magma_int_cast(Q.size(-2), "Q.size(-2)");
  magma_int_t n = magma_int_cast(R.size(-1), "R.size(-1)");
  magma_int_t k = m < n ? m : n;
  magma_int_t nb = magmaGeqrfOptimalBlocksize<scalar_t>(m, n);
  int64_t batch_size = batchCount(R);

  // magmaGeqrf uses a hybrid CPU-GPU algorithm to compute the elementary reflectors.
  // The driver routine magma_(d/s)geqrf2_gpu accepts a tensor on the CPU for elementary reflectors.
  Tensor tau = at::empty({k}, Q.options().device(at::kCPU));
  Tensor work = at::empty({(2 * k + magma_roundup(n, 32)) * nb}, R.options());
  scalar_t* tau_data = tau.data_ptr<scalar_t>();
  scalar_t* work_data = work.data_ptr<scalar_t>();

  // This phase computes R (the raw version)
  // This uses MAGMA's ?geqrf2_gpu function
  magma_int_t info = 0;
  for (int64_t i = 0; i < batch_size; i++) {
    scalar_t* r_working_ptr = &r_data[i * r_matrix_stride];
    magmaGeqrf<scalar_t>(m, n, r_working_ptr, m, tau_data, work_data, &info, /*is_v2=*/true);
    infos[i] = info;
    if (info != 0) {
      return;
    }
  }

  // This phase computes Q (the raw version)
  // We require to perform ?geqrf_gpu again due to this bug in MAGMA:
  // - ?geqrf_gpu allows fast computation of Q via ?orgqr_gpu, but doesn't give R properly.
  // - ?geqrf2_gpu gives correct R, but doesn't allow computation of Q via ?orgqr_gpu
  // Refer to the below link for more details:
  // http://icl.cs.utk.edu/magma/forum/viewtopic.php?f=2&t=1015&p=2800&hilit=geqrf_gpu#p2800
  for (int64_t i = 0; i < batch_size; i++) {
    scalar_t* q_working_ptr = &q_data[i * q_matrix_stride];
    magmaGeqrf<scalar_t>(m, n, q_working_ptr, m, tau_data, work_data, &info, /*is_v2=*/false);
    infos[i] = info;
    if (info != 0) {
      return;
    }
    magmaOrgqr<scalar_t>(m, n_columns, k, q_working_ptr, m, tau_data, work_data, nb, &info);
    infos[i] = info;
    if (info != 0) {
      return;
    }
  }
#endif
}

std::tuple<Tensor,Tensor> _qr_helper_cuda(const Tensor& self, bool some) {
  std::vector<int64_t> infos(batchCount(self), 0);

  // Setup input geometry and inputs for apply_qr
  std::vector<int64_t> q_sizes, q_strides;
  int64_t n_columns_q;
  std::tie(q_sizes, q_strides, n_columns_q) = _compute_geometry_for_Q(self, some);
  Tensor q_working_copy, r_working_copy;

  // If there are no elements, then we simply return a pair of tensors of required dimensions
  if (self.numel() == 0) {
    // Fix the number of columns of q_working_copy appropriately
    q_sizes[self.dim() - 1] = n_columns_q;
    q_working_copy = at::eye(q_sizes[self.dim() - 2], q_sizes[self.dim() - 1], self.options());
    q_working_copy = q_working_copy.expand_as(q_working_copy);

    // We repurpose the same q_sizes for r_working_copy
    // Fix the number of rows and columns of q_working_copy appropriately
    q_sizes[self.dim() - 1] = self.size(-1);
    q_sizes[self.dim() - 2] = n_columns_q;
    r_working_copy = at::empty(q_sizes, self.options());
    return std::make_tuple(q_working_copy, r_working_copy);
  }

  q_working_copy = at::empty_strided(q_sizes, q_strides, self.options());
  q_working_copy.narrow(-1, 0, self.size(-1)).copy_(self);
  r_working_copy = cloneBatchedColumnMajor(self);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "qr_cuda", [&]{
    apply_qr<scalar_t>(q_working_copy, r_working_copy, n_columns_q, infos);
  });
  if (self.dim() > 2) {
    batchCheckErrors(infos, "qr_cuda");
  } else {
    singleCheckErrors(infos[0], "qr_cuda");
  }

  return std::make_tuple(q_working_copy.narrow(-1, 0, n_columns_q),
                         r_working_copy.narrow(-2, 0, n_columns_q).triu());
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ symeig ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_symeig(Tensor& self, Tensor& eigvals, bool eigenvectors, bool upper, std::vector<int64_t>& infos) {
#ifndef USE_MAGMA
AT_ERROR("symeig: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto eigvals_data = eigvals.data_ptr<value_t>();
  auto self_matrix_stride = matrixStride(self);
  auto eigvals_stride = eigvals.size(-1);
  int64_t batch_size = batchCount(self);
  magma_int_t n = magma_int_cast(self.size(-1), "n");

  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;
  magma_vec_t jobz = eigenvectors ? MagmaVec : MagmaNoVec;

  scalar_t* wA;
  ALLOCATE_ARRAY(wA, scalar_t, n * n);

  magma_int_t info;
  // Run once, first to get the optimum work sizes.
  // Since we deal with batches of matrices with the same dimensions, doing this outside
  // the loop saves (batch_size - 1) workspace queries which would provide the same result
  // and (batch_size - 1) calls to allocate and deallocate workspace using at::empty()
  magma_int_t lwork = -1;
  scalar_t wkopt;
  magma_int_t liwork = -1;
  magma_int_t iwkopt;
  magma_int_t lrwork = -1;
  value_t rwkopt;
  magmaSymeig<scalar_t, value_t>(jobz, uplo, n, self_data, n, eigvals_data, wA, n, &wkopt, lwork, &rwkopt, lrwork, &iwkopt, liwork, &info);

  scalar_t* work;
  magma_int_t* iwork;
  lwork = magma_int_cast(real_impl<scalar_t, value_t>(wkopt), "work_size");
  liwork = magma_int_cast(iwkopt, "iwork_size");
  ALLOCATE_ARRAY(work, scalar_t, lwork);
  ALLOCATE_ARRAY(iwork, magma_int_t, liwork);

  value_t* rwork = nullptr;
  c10::Storage storage_rwork;
  if (isComplexType(at::typeMetaToScalarType(self.dtype()))) {
    lrwork = magma_int_cast(rwkopt, "rwork_size");
    storage_rwork = pin_memory<value_t>(lrwork);
    rwork = static_cast<value_t*>(storage_rwork.data());
  }

  for (int64_t i = 0; i < batch_size; i++) {
    scalar_t* self_working_ptr = &self_data[i * self_matrix_stride];
    value_t* eigvals_working_ptr = &eigvals_data[i * eigvals_stride];
    magmaSymeig<scalar_t, value_t>(jobz, uplo, n, self_working_ptr, n, eigvals_working_ptr,
                          wA, n, work, lwork, rwork, lrwork, iwork, liwork, &info);
    infos[i] = info;
    if (info != 0) {
      return;
    }
  }
#endif
}

std::tuple<Tensor, Tensor> _symeig_helper_cuda(const Tensor& self, bool eigenvectors, bool upper) {
  std::vector<int64_t> infos(batchCount(self), 0);

  auto self_sizes = self.sizes().vec();
  self_sizes.pop_back();
  ScalarType dtype = toValueType(typeMetaToScalarType(self.dtype()));

  // magmaSymeig uses a hybrid CPU-GPU algorithm to compute the eigenvalues and eigenvectors.
  // The driver routine magma_(d/s)syev_gpu accepts a tensor on the CPU for eigvalenvalues.
  // The data is later moved to the appropriate device.
  // In the case where self.numel() == 0, we just return an empty tensor of
  // dimensions on the CUDA (to avoid the unnecessary "to(at::kCUDA)")
  auto eigvals_working_copy = self.numel() == 0
                              ? at::empty(self_sizes, self.options().dtype(dtype))
                              : at::empty(self_sizes, self.options().dtype(dtype).device(at::kCPU));

  if (self.numel() == 0) {
    return std::tuple<Tensor, Tensor>(eigvals_working_copy, at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  }

  auto self_working_copy = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "symeig_cuda", [&]{
    apply_symeig<scalar_t>(self_working_copy, eigvals_working_copy, eigenvectors, upper, infos);
  });

  if (self.dim() > 2) {
    batchCheckErrors(infos, "symeig_cuda");
  } else {
    singleCheckErrors(infos[0], "symeig_cuda");
  }
  if (eigenvectors) {
    return std::tuple<Tensor, Tensor>(eigvals_working_copy.to(self.device()), self_working_copy);
  } else {
    return std::tuple<Tensor, Tensor>(eigvals_working_copy.to(self.device()), at::empty({0}, self.options()));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ eig ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// magmaEig uses a hybrid CPU-GPU algorithm, which takes and return CPU
// memory. So, we accept a GPU tensor, copy it to CPU memory, and later copy
// the returned values from CPU to GPU. See also magmaSymeig, which uses a
// similar approach.

template <typename scalar_t>
static void apply_eig(const Tensor& self, bool eigenvectors, Tensor& out_eigvals, Tensor& out_eigvecs,
                      int64_t *info_ptr) {
#ifndef USE_MAGMA
TORCH_CHECK(false, "Calling torch.eig on a CUDA tensor requires compiling PyTorch with MAGMA. "
                   "Either transfer the tensor to the CPU before calling torch.eig or recompile with MAGMA.");
#else
  TORCH_INTERNAL_ASSERT(self.device() == at::kCPU, "Internal error: apply_eig needs a CPU tensor");
  magma_vec_t jobvr = eigenvectors ? MagmaVec : MagmaNoVec;
  magma_int_t n = magma_int_cast(self.size(-1), "n");
  auto self_data = self.data_ptr<scalar_t>();

  auto out_eigvals_data = out_eigvals.data_ptr<scalar_t>();
  scalar_t *wr = out_eigvals_data;
  scalar_t *wi = out_eigvals_data+n;

  scalar_t *vr_data = NULL;
  magma_int_t ldvr = 1;
  if (jobvr == MagmaVec)
  {
      vr_data = out_eigvecs.data_ptr<scalar_t>();
      ldvr = n;
  }

  if (n > 0) {
    // call magmaEig once to get the optimal size of work_data
    scalar_t wkopt;
    magma_int_t info;
    magmaEig<scalar_t>(MagmaNoVec, jobvr, n, self_data, n, wr, wi, NULL, 1, vr_data, ldvr, &wkopt, -1, &info);
    magma_int_t lwork = (magma_int_t) wkopt;

    // call it a 2nd time to to the actual work
    scalar_t *work_data = nullptr;
    ALLOCATE_ARRAY(work_data, scalar_t, lwork);
    magmaEig<scalar_t>(MagmaNoVec, jobvr, n, self_data, n, wr, wi, NULL, 1, vr_data, ldvr, work_data, lwork, &info);
    *info_ptr = info;
  }
#endif
}

/*
 * Internal helper; like eig_cuda but:
 *   1. assume that self is a square matrix of side "n"
 *   2. return CPU tensors (because this is what magmaEig returns), which will be copied to GPU memory
 *      by the caller
 */
std::tuple<Tensor, Tensor> eig_kernel_impl(const Tensor& self, bool& eigenvectors) {
  int64_t n = self.size(-1);
  // copy self to pinned CPU memory
  auto self_working_copy = at::empty_strided(
      {n, n}, // square matrix
      {1, n}, // column-ordered, as magmaEig expects
      at::TensorOptions(at::kCPU).dtype(self.dtype()).pinned_memory(true));
  self_working_copy.copy_(self);

  // tensors holding the results. We use empty_strided to make them column-ordered
  auto options = self.options().device(at::kCPU).memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto out_eigvals = at::empty_strided({n, 2}, {1, n}, options);
  auto out_eigvecs = eigenvectors
                     ? at::empty_strided({n, n}, {1, n}, options)
                     : Tensor();

  int64_t info;
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "eig_cuda", [&]{
    apply_eig<scalar_t>(self_working_copy, eigenvectors, out_eigvals, out_eigvecs, &info);
  });
  singleCheckErrors(info, "eig_cuda");

  return std::tuple<Tensor, Tensor>(out_eigvals, out_eigvecs);
}

REGISTER_DISPATCH(eig_stub, &eig_kernel_impl);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ syevd ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// This function computes eigenvalues 'w' and eigenvectors 'v' of the tensor 'self'
// compute_eigenvectors controls whether eigenvectors should be computed
// uplo controls the portion of input matrix to consider in computations, allowed values are "u", "U", "l", "L"
// '_symeig_helper_cuda' prepares correct input for 'apply_symeig' and checks for possible errors using 'infos'
// See also CPU implementation in aten/src/ATen/native/BatchLinearAlgebra.cpp
std::tuple<Tensor, Tensor> _syevd_helper_cuda(const Tensor& self, bool compute_eigenvectors, std::string uplo_str) {
  // NumPy allows lowercase input for UPLO argument
  // It is assumed that uplo_str is either "U" or "L"
  char uplo = std::toupper(uplo_str[0]);
  bool upper = uplo == 'U' ? true : false;
  return _symeig_helper_cuda(self, compute_eigenvectors, upper);
}
    
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ svd ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<typename scalar_t>
static void apply_svd(Tensor& self, Tensor& U, Tensor& S, Tensor& VT,
                      char jobchar, std::vector<int64_t>& infos) {
#ifndef USE_MAGMA
AT_ERROR("svd: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto U_data = U.data_ptr<scalar_t>();
  auto S_data = S.data_ptr<value_t>();
  auto VT_data = VT.data_ptr<scalar_t>();
  auto self_stride = matrixStride(self);
  auto U_stride = matrixStride(U);
  auto S_stride = S.size(-1);
  auto VT_stride = matrixStride(VT);
  auto batchsize = batchCount(self);

  magma_vec_t jobz = jobchar == 'A' ? MagmaAllVec : (jobchar == 'S' ? MagmaSomeVec : MagmaNoVec);

  magma_int_t m = magma_int_cast(self.size(-2), "m");
  magma_int_t n = magma_int_cast(self.size(-1), "n");
  auto mn = std::min(m, n);

  c10::Storage storage_rwork;
  value_t* rwork = nullptr;

  magma_int_t* iwork;
  ALLOCATE_ARRAY(iwork, magma_int_t, 8 * mn);
  if (isComplexType(at::typeMetaToScalarType(self.dtype()))) {
    auto lrwork = computeLRWorkDim(jobchar, m, n);
    storage_rwork = pin_memory<value_t>(lrwork);
    rwork = static_cast<value_t*>(storage_rwork.data());
  }

  magma_int_t info = 0;
  // Run once, first to get the optimum work size.
  // Since we deal with batches of matrices with the same dimensions, doing this outside
  // the loop saves (batch_size - 1) workspace queries which would provide the same result
  // and (batch_size - 1) calls to allocate and deallocate workspace using at::empty()
  magma_int_t lwork = -1;
  scalar_t wkopt;
  magmaSvd<scalar_t, value_t>(jobz, m, n, self_data, m, S_data, U_data, m, VT_data, n, &wkopt, lwork, rwork, iwork, &info);
  lwork = magma_int_cast(real_impl<scalar_t, value_t>(wkopt), "work_size");
  scalar_t* work;
  ALLOCATE_ARRAY(work, scalar_t, lwork);

  for (int64_t i = 0; i < batchsize; i++) {
    scalar_t* self_working_ptr = &self_data[i * self_stride];
    value_t* S_working_ptr = &S_data[i * S_stride];
    scalar_t* U_working_ptr = &U_data[i * U_stride];
    scalar_t* VT_working_ptr = &VT_data[i * VT_stride];

    // Compute S, U (optionally), VT (optionally)
    magmaSvd<scalar_t, value_t>(jobz, m, n, self_working_ptr, m,
                                S_working_ptr, U_working_ptr, m, VT_working_ptr, n, work, lwork, rwork, iwork, &info);
    infos[i] = info;
    if (info != 0) {
      return;
    }
  }
#endif
}

std::tuple<Tensor, Tensor, Tensor> _svd_helper_cuda(const Tensor& self, bool some, bool compute_uv) {
  std::vector<int64_t> infos(batchCount(self), 0);
  int64_t m = self.size(-2), n = self.size(-1);
  int64_t k = std::min(m, n);

  char jobchar = compute_uv ? (some ? 'S' : 'A') : 'N';

  Tensor U_working_copy, S_working_copy, VT_working_copy;
  std::tie(U_working_copy, S_working_copy, VT_working_copy) = _create_U_S_VT(self, some, compute_uv);

  if (self.numel() > 0) {
    // The input matrix, U, S and VT have to reside in pinned memory.
    // Additionally, the input and U have to be in column major format.
    // _create_U_S_VT takes care of a part of these requirements (for U, S and VT)
    // For the input matrix, this requirements are being taken care of below.
    // Specify strides
    auto self_col_major_strides = at::detail::defaultStrides(self.sizes());
    self_col_major_strides[self.dim() - 2] = 1;
    self_col_major_strides[self.dim() - 1] = m;
    // Create strided tensor in pinned memory
    auto self_working_copy = at::empty_strided(self.sizes(), self_col_major_strides,
                                               at::TensorOptions(at::kCPU).dtype(self.dtype()).pinned_memory(true));
    self_working_copy.copy_(self);

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "svd_cuda", [&] {
      apply_svd<scalar_t>(self_working_copy, U_working_copy, S_working_copy, VT_working_copy, jobchar, infos);
    });

    if (self.dim() > 2) {
      batchCheckErrors(infos, "svd_cuda");
    } else {
      singleCheckErrors(infos[0], "svd_cuda");
    }

    U_working_copy = same_stride_to(U_working_copy, self.options());
    S_working_copy = same_stride_to(S_working_copy, S_working_copy.options().device(self.device()));
    VT_working_copy = same_stride_to(VT_working_copy, self.options());

    if (compute_uv) {
      if (some) {
        VT_working_copy = VT_working_copy.narrow(-1, 0, k);
      }
    } else {
      VT_working_copy.zero_();
      U_working_copy.zero_();
    }
  } else {
    U_working_copy = same_stride_to(U_working_copy, self.options()).zero_();
    S_working_copy = same_stride_to(S_working_copy, S_working_copy.options().device(self.device()));
    VT_working_copy = same_stride_to(VT_working_copy, self.options()).zero_();
  }
  return std::make_tuple(U_working_copy, S_working_copy, VT_working_copy);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_lu_solve(Tensor& b, const Tensor& lu, const Tensor& pivots, int64_t& info) {
#ifndef USE_MAGMA
AT_ERROR("lu_solve: MAGMA library not found in "
         "compilation. Please rebuild with MAGMA.");
#else
  auto b_data = b.data_ptr<scalar_t>();
  auto lu_data = lu.data_ptr<scalar_t>();

  auto n = lu.size(-2);
  auto nrhs = b.size(-1);

  int info_tmp = 0;
  if (b.dim() == 2) {
    Tensor pivots_tmp = pivots.cpu();
    magmaLuSolve<scalar_t>(n, nrhs, lu_data, n, pivots_tmp.data_ptr<magma_int_t>(), b_data, n, &info_tmp);
    info = info_tmp;
  } else {
    auto pivots_data = pivots.data_ptr<magma_int_t>();

    auto b_stride = matrixStride(b);
    auto lu_stride = matrixStride(lu);
    auto pivots_stride = pivots.size(-1);
    magma_int_t batch_size = magma_int_cast(batchCount(b), "batchCount");

    magma_int_t** pivots_array;
    scalar_t** lu_array;
    scalar_t** b_array;

    ALLOCATE_ARRAY(pivots_array, magma_int_t*, batch_size);
    ALLOCATE_ARRAY(lu_array, scalar_t*, batch_size);
    ALLOCATE_ARRAY(b_array, scalar_t*, batch_size);

    for (int64_t i = 0; i < batch_size; i++) {
      pivots_array[i] = &pivots_data[i * pivots_stride];
      b_array[i] = &b_data[i * b_stride];
      lu_array[i] = &lu_data[i * lu_stride];
    }

    MAGMAQueue magma_queue(b.get_device());

    constexpr int64_t batch_limit = 65535;
    // Compute as many batches of 65535 possible
    // The number of "mini"-batches are floor(batch_size / batch_limit)
    // and these cover floor(batch_size / batch_limit) * batch_limit matrix solves
    int64_t mini_batches = batch_size / batch_limit, mini_idx;
    for (mini_idx = 0; mini_idx < mini_batches * batch_limit; mini_idx += batch_limit) {
      scalar_t** lu_array_cur = &lu_array[mini_idx];
      scalar_t** b_array_cur = &b_array[mini_idx];
      magma_int_t** pivots_array_cur = &pivots_array[mini_idx];

      magmaLuSolveBatched<scalar_t>(
          n, nrhs, lu_array_cur, n, pivots_array_cur, b_array_cur, n,
          info_tmp, batch_limit, magma_queue);

      if (info_tmp != 0) {
        break;
      }
    }

    // Compute whatever is left = batch_size - floor(batch_size / batch_limit) * batch_limit
    // which concisely is equal to batch_size % batch_limit
    if (batch_size % batch_limit != 0 && info_tmp == 0) {
      magmaLuSolveBatched<scalar_t>(
          n, nrhs, &lu_array[mini_idx], n, &pivots_array[mini_idx], &b_array[mini_idx], n,
          info_tmp, batch_size % batch_limit, magma_queue);
    }

    info = info_tmp;
  }
#endif
}

Tensor _lu_solve_helper_cuda(const Tensor& self, const Tensor& LU_data, const Tensor& LU_pivots) {
  int64_t info = 0;
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto LU_data_working_copy = cloneBatchedColumnMajor(LU_data);
  auto LU_pivots_working_copy = LU_pivots.is_contiguous() ? LU_pivots : LU_pivots.contiguous();

  if (self.numel() == 0 || LU_data.numel() == 0) {
    return at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "lu_solve_cuda", [&]{
    apply_lu_solve<scalar_t>(self_working_copy, LU_data_working_copy, LU_pivots_working_copy, info);
  });
  TORCH_CHECK(info == 0, "MAGMA lu_solve : invalid argument: ", -info);
  return self_working_copy;
}

}}  // namespace at::native

#undef ALLOCATE_ARRAY
