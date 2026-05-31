#include "algorithms/ohm_miner.h"
extern "C" {
#include "core/dm_benchmark.h"
#include "core/dm_dataset_types.h"
#include "algorithms/sparc_hoi.h"
}

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>

struct Node {
    uint32_t item;
    std::vector<uint32_t> tids;
    double sum_inv;
};

struct OhmCtx {
    size_t min_support;
    double threshold;
    bool summed_mode;
    std::vector<double> global_inv_len;
    
    size_t raw_count;
    size_t total_output_items;
    size_t intersection_tests;
    
    size_t max_patterns;
    double max_seconds;
    std::chrono::high_resolution_clock::time_point start_time;
    bool limited;
};

static bool should_stop(OhmCtx& ctx) {
    if (ctx.max_patterns && ctx.raw_count >= ctx.max_patterns) {
        ctx.limited = true;
        return true;
    }
    if (ctx.max_seconds > 0.0) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - ctx.start_time).count();
        if (elapsed >= ctx.max_seconds) {
            ctx.limited = true;
            return true;
        }
    }
    return false;
}

static void intersect_nodes(const Node& A, const Node& B, Node& out, OhmCtx& ctx) {
    ctx.intersection_tests++;
    out.item = B.item;
    out.sum_inv = 0.0;
    
    size_t i = 0, j = 0;
    size_t sizeA = A.tids.size();
    size_t sizeB = B.tids.size();
    
    // Reserve capacity to avoid reallocations
    out.tids.reserve(std::min(sizeA, sizeB));
    
    while (i < sizeA && j < sizeB) {
        if (A.tids[i] < B.tids[j]) {
            i++;
        } else if (A.tids[i] > B.tids[j]) {
            j++;
        } else {
            uint32_t tid = A.tids[i];
            out.tids.push_back(tid);
            out.sum_inv += ctx.global_inv_len[tid];
            i++;
            j++;
        }
    }
}

static void mine_recursive(OhmCtx& ctx, const Node& parent, const std::vector<Node>& siblings, size_t depth) {
    if (should_stop(ctx)) return;

    std::vector<Node> children;
    children.reserve(siblings.size());

    for (size_t i = 0; i < siblings.size(); ++i) {
        Node child;
        intersect_nodes(parent, siblings[i], child, ctx);
        
        if (child.tids.size() < ctx.min_support) continue;
        
        double occ = 0.0;
        if (ctx.summed_mode) {
            occ = (double)(depth + 1) * child.sum_inv;
        } else {
            occ = ((double)(depth + 1) * child.sum_inv) / (double)child.tids.size();
        }
        
        if (occ + 1e-12 >= ctx.threshold) {
            ctx.raw_count++;
            ctx.total_output_items += depth + 1;
        }
        
        children.push_back(std::move(child));
    }

    for (size_t i = 0; i < children.size(); ++i) {
        double max_possible_occ = 0.0;
        size_t remaining = children.size() - i - 1;
        if (ctx.summed_mode) {
            max_possible_occ = (double)(depth + 1 + remaining) * children[i].sum_inv;
        } else {
            max_possible_occ = ((double)(depth + 1 + remaining) * children[i].sum_inv) / (double)ctx.min_support;
        }
        
        if (max_possible_occ + 1e-12 >= ctx.threshold) {
            std::vector<Node> next_siblings(children.begin() + i + 1, children.end());
            mine_recursive(ctx, children[i], next_siblings, depth + 1);
        }
    }
}

extern "C" {

static DM_Status ohm_miner_run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_TRANSACTIONAL || !ds->payload) return DM_ERROR_INCOMPATIBLE;
    
    DM_SPARC_HOI_Params *p = (DM_SPARC_HOI_Params *)params;
    OhmCtx ctx;
    ctx.raw_count = 0;
    ctx.total_output_items = 0;
    ctx.intersection_tests = 0;
    ctx.start_time = std::chrono::high_resolution_clock::now();
    ctx.limited = false;
    
    size_t ntrans = ds->count;
    DM_Trans_Simple* src = (DM_Trans_Simple*)ds->payload;
    
    double min_occupancy = p ? p->min_occupancy : 0.5;
    ctx.summed_mode = p ? p->summed_occupancy_mode : false;
    ctx.threshold = ctx.summed_mode ? ((min_occupancy < 1.0) ? min_occupancy * (double)ntrans : min_occupancy) : min_occupancy;
    ctx.min_support = (p && p->min_support) ? p->min_support : (size_t)std::ceil(min_occupancy * (double)ntrans);
    if (ctx.min_support < 1) ctx.min_support = 1;
    
    ctx.max_patterns = p ? p->max_patterns : 0;
    ctx.max_seconds = p ? p->max_seconds : 0.0;
    
    ctx.global_inv_len.resize(ntrans, 0.0);
    
    std::vector<Node> root_nodes(ds->max_id + 1);
    for (uint32_t i = 0; i <= ds->max_id; ++i) {
        root_nodes[i].item = i;
        root_nodes[i].sum_inv = 0.0;
    }
    
    for (size_t t = 0; t < ntrans; ++t) {
        double inv = src[t].count ? 1.0 / (double)src[t].count : 0.0;
        ctx.global_inv_len[t] = inv;
        
        for (size_t i = 0; i < src[t].count; ++i) {
            uint32_t item = src[t].items[i];
            root_nodes[item].tids.push_back((uint32_t)t);
            root_nodes[item].sum_inv += inv;
        }
    }
    
    std::vector<Node> active_nodes;
    for (uint32_t i = 0; i <= ds->max_id; ++i) {
        if (root_nodes[i].tids.size() >= ctx.min_support) {
            active_nodes.push_back(std::move(root_nodes[i]));
        }
    }
    
    // Sort items by support descending (typical heuristic for Eclat / HOI)
    std::sort(active_nodes.begin(), active_nodes.end(), [](const Node& a, const Node& b) {
        if (a.tids.size() != b.tids.size()) return a.tids.size() < b.tids.size();
        return a.item < b.item;
    });

    std::cout << "[Ohm-Miner] Bắt đầu duyệt Vertical Quantum State Intersection...\n";
    std::cout << "[Ohm-Miner] active_items=" << active_nodes.size() 
              << " minsup=" << ctx.min_support 
              << " threshold=" << ctx.threshold << "\n";
              
    for (size_t i = 0; i < active_nodes.size(); ++i) {
        if (should_stop(ctx)) break;
        
        double occ = 0.0;
        if (ctx.summed_mode) occ = active_nodes[i].sum_inv;
        else occ = active_nodes[i].sum_inv / (double)active_nodes[i].tids.size();
        
        if (occ + 1e-12 >= ctx.threshold) {
            ctx.raw_count++;
            ctx.total_output_items += 1;
        }
        
        std::vector<Node> siblings(active_nodes.begin() + i + 1, active_nodes.end());
        mine_recursive(ctx, active_nodes[i], siblings, 1);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end_time - ctx.start_time).count();
    
    std::cout << "[Ohm-Miner] Complete. Total Patterns: " << ctx.raw_count << "\n";
    std::cout << "[Ohm-Miner] Phép giao thoa (Intersections): " << ctx.intersection_tests << " lần.\n";
    std::cout << "[Ohm-Miner] Time: " << seconds << "s. Limited: " << (ctx.limited ? "Yes" : "No") << "\n";
    
    dm_bench_record_results(ctx.raw_count, ctx.total_output_items);
    
    return DM_SUCCESS;
}

DM_Algorithm ohm_miner_algo = {
    "ohm_miner",
    "Ohm-Miner (Quantum State Intersection)",
    "Exact High Occupancy Itemset Mining via TID-List Wavefunction Intersections.",
    (1 << DM_TYPE_TRANSACTIONAL),
    ohm_miner_run
};

DM_REGISTER_ALGORITHM(ohm_miner_algo)

} // extern "C"
