/**
 * Parallax algorithm
 *
 * Nodescape 2018 - GPL code
 */

#include <stdio.h>
#include <memory.h>
#include <unistd.h>

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"

#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x16.h"

static uint32_t *d_hash[MAX_GPUS];
// static __thread uint32_t s_ntime = UINT32_MAX;
static __thread int pViewPoint;
static __thread int pView[9]; 

static void getView(const uint32_t* prevblock, int *output)
{
	// Views:
	int nView[16][9] = {
		{ 0, 1, 2, 3, 4, 5, 6, 7, 8 },
		{ 0, 1, 2, 3, 5, 4, 6, 7, 8 },
		{ 0, 1, 2, 5, 3, 6, 4, 7, 8 },
		{ 0, 1, 5, 2, 6, 3, 7, 4, 8 },
		{ 0, 5, 1, 6, 2, 7, 3, 8, 4 },
		{ 5, 0, 6, 1, 7, 2, 8, 3, 4 },
		{ 5, 6, 0, 7, 1, 8, 2, 3, 4 },
		{ 5, 6, 7, 0, 8, 1, 2, 3, 4 },
		{ 5, 6, 7, 8, 0, 1, 2, 3, 4 },
		{ 5, 6, 7, 0, 8, 1, 2, 3, 4 },
		{ 5, 6, 0, 7, 1, 8, 2, 3, 4 },
		{ 5, 0, 6, 1, 7, 2, 8, 3, 4 },
		{ 0, 5, 1, 6, 2, 7, 3, 8, 4 },
		{ 0, 1, 5, 2, 6, 3, 7, 4, 8 },
		{ 0, 1, 2, 5, 3, 6, 4, 7, 8 },
		{ 0, 1, 2, 3, 5, 4, 6, 7, 8 }
	};

	uint8_t* data = (uint8_t*)prevblock;
	int sView = data[0] & 0xF;
	for (uint8_t i = 0; i < 9; i++) {
		output[i] = nView[sView][i];
	}
}

// Parallax CPU Hash (Validation)
extern "C" void parallax_hash(void *output, const void *input)
{
	unsigned char _ALIGN(128) hash[128] = { 0 };

    static unsigned char pblank[1];

    sph_skein512_context     ctx_skein;     // 0
    sph_luffa512_context     ctx_luffa;     // 1
    sph_keccak512_context    ctx_keccak;    // 2
    sph_jh512_context        ctx_jh;        // 3
    sph_groestl512_context   ctx_groestl;   // 4
    sph_echo512_context      ctx_echo;      // 5
    sph_cubehash512_context  ctx_cubehash;  // 6
    sph_bmw512_context       ctx_bmw;       // 7
    sph_blake512_context     ctx_blake;     // 8      
    
    void *in = (void*) input;
	int size = 80;
	uint32_t *in32 = (uint32_t*) input;
    getView(&in32[1], pView);

	for (int i = 0; i < 9; ++i)
    {
		switch (pView[i]) {
			case 0:
				sph_skein512_init(&ctx_skein);
				sph_skein512(&ctx_skein, in, size);
				sph_skein512_close(&ctx_skein, hash);
				break;
			case 1:
				sph_luffa512_init(&ctx_luffa);
				sph_luffa512(&ctx_luffa, in, size);
				sph_luffa512_close(&ctx_luffa, hash);
				break;
			case 2:
				sph_keccak512_init(&ctx_keccak);
				sph_keccak512(&ctx_keccak, in, size);
				sph_keccak512_close(&ctx_keccak, hash);
				break;
			case 3:
				sph_jh512_init(&ctx_jh);
				sph_jh512(&ctx_jh, in, size);
				sph_jh512_close(&ctx_jh, hash);
				break;
			case 4:
				sph_groestl512_init(&ctx_groestl);
				sph_groestl512(&ctx_groestl, in, size);
				sph_groestl512_close(&ctx_groestl, hash);
				break;
			case 5:
				sph_echo512_init(&ctx_echo);
				sph_echo512(&ctx_echo, in, size);
				sph_echo512_close(&ctx_echo, hash);
				break;
			case 6:
				sph_cubehash512_init(&ctx_cubehash);
				sph_cubehash512(&ctx_cubehash, in, size);
				sph_cubehash512_close(&ctx_cubehash, hash);
				break;
			case 7:
				sph_bmw512_init(&ctx_bmw);
				sph_bmw512(&ctx_bmw, in, size);
				sph_bmw512_close(&ctx_bmw, hash);
				break;
			case 8:
				sph_blake512_init(&ctx_blake);
				sph_blake512(&ctx_blake, in, size);
				sph_blake512_close(&ctx_blake, hash);
				break;
		}
		in = (void*) hash;
		size = 64;
	}	
	memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

//#define _DEBUG
#define _DEBUG_PREFIX "parallax-"
#include "cuda_debug.cuh"

extern "C" int scanhash_parallax(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 20 : 19;
	if (strstr(device_name[dev_id], "GTX 1080")) intensity = 20;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);
		use_compat_kernels[thr_id] = (cuda_arch[dev_id] < 500);
		if (use_compat_kernels[thr_id])
			x11_echo512_cpu_init(thr_id, throughput);
		quark_blake512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		qubit_luffa512_cpu_init(thr_id, throughput);
		x11_luffa512_cpu_init(thr_id, throughput); // 64
		x11_shavite512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput); // 64
		x16_echo512_cuda_init(thr_id, throughput);;

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), 0);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	if (opt_benchmark) {
		((uint32_t*)ptarget)[7] = 0x003f;
		((uint8_t*)pdata)[8] = 0xAA; // hashOrder[0] = 'A'; for echo 80 + 64
	}

	uint32_t _ALIGN(64) endiandata[20];
	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);
	// uint32_t ntime = swab32(pdata[17]);
	getView(&endiandata[1], pView);

	if (opt_debug && !thr_id) applog(LOG_DEBUG, "View %i, %i, %i, %i, %i, %i, %i, %i, %i", pView[0], pView[1], pView[2], pView[3], pView[4], pView[5], pView[6], pView[7], pView[8]);

	cuda_check_cpu_setTarget(ptarget);

	switch (pView[0]) {
		case 0:
			skein512_cpu_setBlock_80((void*)endiandata);
			break;
		case 5:
			x16_echo512_setBlock_80((void*)endiandata);
			break;
		default: {
			return -1;
		}
	}

	do {
		int order = 0;

		// Hash with CUDA

		switch (pView[0]) {
			case 0:
				skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], 1); order++;
				TRACE("skein80:");
				break;
			case 5:
				x16_echo512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("echo   :");
				break;
		}

		for (int i = 1; i < 9; i++)
		{
			switch (pView[i]) {
				case 0:
					quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
					TRACE("skein  :");
					break;
				case 1:
					x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
					TRACE("luffa  :");
					break;
				case 2:
					quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
					TRACE("keccak :");
					break;
				case 3:
					quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
					TRACE("jh512  :");
					break;
				case 4:
					quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
					TRACE("groestl:");
					break;
				case 5:
					if (use_compat_kernels[thr_id])
						x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
					else {
						x16_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
					}
					TRACE("echo   :");
					break;
				case 6:
					x11_cubehash512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
					TRACE("cube   :");
					break;
				case 7:
					quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
					TRACE("bmw    :");
					break;
				case 8:
					quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
					TRACE("blake  :");
					break;
			}
		}

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);

#ifdef _DEBUG
		uint32_t _ALIGN(64) dhash[8];
		be32enc(&endiandata[19], pdata[19]);
		parallax_hash(dhash, endiandata);
		applog_hash(dhash);
		return -1;
#endif
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			parallax_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					parallax_hash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}

				return work->valid_nonces;
			} else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_parallax(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
