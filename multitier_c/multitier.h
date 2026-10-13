#ifndef MULTITIER_H
#define MULTITIER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

typedef struct {
    char magic[4];          /* Magic bytes: GGUF */
    uint32_t version;       /* GGUF version (3) */
    uint64_t tensor_count;  /* Number of tensors in GGUF file */
    uint64_t metadata_kv_count; /* Number of metadata KV pairs */
} tmr_gguf_header_t;

typedef struct {
    char model_name[128];
    uint32_t d_model;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t num_experts;
    double export_throughput_mbs;
    double load_throughput_mbs;
} tmr_runtime_stats_t;

int tmr_init_runtime(const char* manifest_path);
int tmr_load_gguf_model(const char* gguf_path, tmr_gguf_header_t* out_header);
void tmr_print_runtime_stats(const tmr_runtime_stats_t* stats);
int tmr_hotswap_expert_shard(uint32_t expert_id, const char* shard_binary_path);
int tmr_hdc_associative_lookup(const uint64_t* query_vector, uint32_t dimension_words, uint32_t* out_match_id);

#endif /* MULTITIER_H */
