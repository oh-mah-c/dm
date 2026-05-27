/*
 * NEAT: NeuroEvolution of Augmenting Topologies.
 * Stanley & Miikkulainen, Evolutionary Computation 10(2):99-127, 2002.
 */

#include "algorithms/neat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>

/* =========================================================================
 * RNG (xorshift64 — fast, seedable, no dependencies)
 * ========================================================================= */

static uint64_t s_rng = 6364136223846793005ULL;

static void rng_seed(unsigned long seed) {
    s_rng = seed ? (uint64_t)seed : (uint64_t)time(NULL);
    if (!s_rng) s_rng = 1;
}

static uint64_t rng_u64(void) {
    s_rng ^= s_rng << 13;
    s_rng ^= s_rng >> 7;
    s_rng ^= s_rng << 17;
    return s_rng;
}

/* uniform [0,1) */
static double rng_f(void) { return (double)(rng_u64() >> 11) / 9007199254740992.0; }

/* uniform [-r, r] */
static double rng_sym(double r) { return (rng_f() * 2.0 - 1.0) * r; }

/* integer [0, n) */
static int rng_int(int n) { return (int)(rng_u64() % (uint64_t)n); }

/* =========================================================================
 * Innovation number registry
 * Tracks (in_node, out_node) -> innovation number within one generation.
 * Ensures same structural mutation in same generation gets same innov (Sec 3.2).
 * ========================================================================= */

typedef struct {
    int in_node;
    int out_node;
    int innov;
} InnovEntry;

typedef struct {
    InnovEntry *entries;
    int         n;
    int         cap;
    int         next_innov; /* global counter */
} InnovDB;

static void innovdb_init(InnovDB *db, int start_innov) {
    db->entries = NULL;
    db->n = 0;
    db->cap = 0;
    db->next_innov = start_innov;
}

static void innovdb_free(InnovDB *db) { free(db->entries); }

/* Clear current-generation cache, keeping counter. */
static void innovdb_new_gen(InnovDB *db) { db->n = 0; }

/* Get (or create) innovation number for a connection. */
static int innovdb_get(InnovDB *db, int in_node, int out_node) {
    for (int i = 0; i < db->n; ++i)
        if (db->entries[i].in_node == in_node &&
            db->entries[i].out_node == out_node)
            return db->entries[i].innov;
    if (db->n == db->cap) {
        db->cap = db->cap ? db->cap * 2 : 16;
        db->entries = realloc(db->entries, db->cap * sizeof(InnovEntry));
    }
    db->entries[db->n].in_node  = in_node;
    db->entries[db->n].out_node = out_node;
    db->entries[db->n].innov    = db->next_innov++;
    return db->entries[db->n++].innov;
}

/* =========================================================================
 * Genome helpers
 * ========================================================================= */

static NEAT_Genome *genome_alloc(int n_nodes, int n_conns) {
    NEAT_Genome *g = calloc(1, sizeof(NEAT_Genome));
    g->nodes   = calloc(n_nodes ? n_nodes : 1, sizeof(NEAT_NodeGene));
    g->n_nodes = n_nodes;
    g->conns   = calloc(n_conns ? n_conns : 1, sizeof(NEAT_ConnGene));
    g->n_conns = n_conns;
    return g;
}

void neat_genome_free(NEAT_Genome *g) {
    if (!g) return;
    free(g->nodes);
    free(g->conns);
    free(g);
}

static NEAT_Genome *genome_copy(const NEAT_Genome *src) {
    NEAT_Genome *g = genome_alloc(src->n_nodes, src->n_conns);
    memcpy(g->nodes, src->nodes, src->n_nodes * sizeof(NEAT_NodeGene));
    memcpy(g->conns, src->conns, src->n_conns * sizeof(NEAT_ConnGene));
    g->fitness    = src->fitness;
    g->species_id = src->species_id;
    return g;
}

/* Find a node gene by id; returns index or -1. */
static int genome_find_node(const NEAT_Genome *g, int id) {
    for (int i = 0; i < g->n_nodes; ++i)
        if (g->nodes[i].id == id) return i;
    return -1;
}

/* Add a new node gene. */
static void genome_add_node(NEAT_Genome *g, int id, int type) {
    g->nodes = realloc(g->nodes, (g->n_nodes + 1) * sizeof(NEAT_NodeGene));
    g->nodes[g->n_nodes].id   = id;
    g->nodes[g->n_nodes].type = type;
    ++g->n_nodes;
}

/* Add a new connection gene. */
static void genome_add_conn(NEAT_Genome *g, int in_node, int out_node,
                            double weight, int enabled, int innov) {
    g->conns = realloc(g->conns, (g->n_conns + 1) * sizeof(NEAT_ConnGene));
    NEAT_ConnGene *c = &g->conns[g->n_conns++];
    c->in_node  = in_node;
    c->out_node = out_node;
    c->weight   = weight;
    c->enabled  = enabled;
    c->innov    = innov;
}

/* Check whether a connection (in,out) already exists. */
static int genome_conn_exists(const NEAT_Genome *g, int in_node, int out_node) {
    for (int i = 0; i < g->n_conns; ++i)
        if (g->conns[i].in_node == in_node && g->conns[i].out_node == out_node)
            return 1;
    return 0;
}

/* Maximum node id across the genome. */
static int genome_max_node_id(const NEAT_Genome *g) {
    int mx = 0;
    for (int i = 0; i < g->n_nodes; ++i)
        if (g->nodes[i].id > mx) mx = g->nodes[i].id;
    return mx;
}

/* =========================================================================
 * Compatibility distance (Eq. 1)
 * delta = c1*E/N + c2*D/N + c3*W_bar
 * ========================================================================= */

double neat_compat(const NEAT_Genome *a, const NEAT_Genome *b,
                   double c1, double c2, double c3) {
    int max_a = 0, max_b = 0;
    for (int i = 0; i < a->n_conns; ++i)
        if (a->conns[i].innov > max_a) max_a = a->conns[i].innov;
    for (int i = 0; i < b->n_conns; ++i)
        if (b->conns[i].innov > max_b) max_b = b->conns[i].innov;

    int max_min = (max_a < max_b) ? max_a : max_b;

    int matching = 0, disjoint = 0, excess = 0;
    double wdiff = 0.0;

    for (int i = 0; i < a->n_conns; ++i) {
        int innov = a->conns[i].innov;
        int found = 0;
        for (int j = 0; j < b->n_conns; ++j) {
            if (b->conns[j].innov == innov) {
                matching++;
                wdiff += fabs(a->conns[i].weight - b->conns[j].weight);
                found = 1;
                break;
            }
        }
        if (!found) {
            if (innov <= max_min) disjoint++;
            else                  excess++;
        }
    }
    for (int j = 0; j < b->n_conns; ++j) {
        int innov = b->conns[j].innov;
        int found = 0;
        for (int i = 0; i < a->n_conns; ++i)
            if (a->conns[i].innov == innov) { found = 1; break; }
        if (!found) {
            if (innov <= max_min) disjoint++;
            else                  excess++;
        }
    }

    int N = (a->n_conns > b->n_conns) ? a->n_conns : b->n_conns;
    if (N < 1) N = 1;
    double w_bar = (matching > 0) ? (wdiff / matching) : 0.0;
    return c1 * excess / N + c2 * disjoint / N + c3 * w_bar;
}

/* =========================================================================
 * Mutations
 * ========================================================================= */

/* Mutate connection weights (Section 3.1, Section 4.1). */
static void mutate_weights(NEAT_Genome *g, double p_perturb,
                           double perturb_range, double weight_range) {
    for (int i = 0; i < g->n_conns; ++i) {
        if (rng_f() < p_perturb)
            g->conns[i].weight += rng_sym(perturb_range);
        else
            g->conns[i].weight  = rng_sym(weight_range);
    }
}

/* Add connection mutation (Figure 3, top). */
static void mutate_add_conn(NEAT_Genome *g, InnovDB *db, double weight_range) {
    int tries = 20;
    while (tries-- > 0) {
        int ai = rng_int(g->n_nodes);
        int bi = rng_int(g->n_nodes);
        if (ai == bi) continue;
        int in_id  = g->nodes[ai].id;
        int out_id = g->nodes[bi].id;
        if (g->nodes[bi].type == NEAT_NODE_SENSOR ||
            g->nodes[bi].type == NEAT_NODE_BIAS)   continue;
        if (genome_conn_exists(g, in_id, out_id))  continue;
        int innov = innovdb_get(db, in_id, out_id);
        genome_add_conn(g, in_id, out_id, rng_sym(weight_range), 1, innov);
        return;
    }
}

/* Add node mutation (Figure 3, bottom). */
static void mutate_add_node(NEAT_Genome *g, InnovDB *db) {
    int enabled_count = 0;
    for (int i = 0; i < g->n_conns; ++i)
        if (g->conns[i].enabled) ++enabled_count;
    if (enabled_count == 0) return;

    int pick = rng_int(enabled_count), idx = 0;
    for (int i = 0; i < g->n_conns; ++i) {
        if (!g->conns[i].enabled) continue;
        if (idx++ == pick) {
            g->conns[i].enabled = 0;
            int in_id  = g->conns[i].in_node;
            int out_id = g->conns[i].out_node;
            double old_w = g->conns[i].weight;

            int new_id = genome_max_node_id(g) + 1;
            genome_add_node(g, new_id, NEAT_NODE_HIDDEN);

            int innov1 = innovdb_get(db, in_id,  new_id);
            int innov2 = innovdb_get(db, new_id, out_id);
            genome_add_conn(g, in_id,  new_id, 1.0,   1, innov1);
            genome_add_conn(g, new_id, out_id, old_w, 1, innov2);
            return;
        }
    }
}

/* =========================================================================
 * Crossover (Section 3.2, Figure 4)
 * ========================================================================= */

static NEAT_Genome *crossover(const NEAT_Genome *fitter, const NEAT_Genome *other) {
    NEAT_Genome *child = calloc(1, sizeof(NEAT_Genome));

    /* Node genes: from fitter, plus any hidden nodes from other not in fitter. */
    child->n_nodes = fitter->n_nodes;
    child->nodes   = malloc(fitter->n_nodes * sizeof(NEAT_NodeGene));
    memcpy(child->nodes, fitter->nodes, fitter->n_nodes * sizeof(NEAT_NodeGene));

    int cap = fitter->n_conns + other->n_conns;
    child->conns   = malloc((cap ? cap : 1) * sizeof(NEAT_ConnGene));
    child->n_conns = 0;

    for (int i = 0; i < fitter->n_conns; ++i) {
        const NEAT_ConnGene *fc = &fitter->conns[i];
        int found = 0;
        for (int j = 0; j < other->n_conns; ++j) {
            if (other->conns[j].innov == fc->innov) {
                /* Matching gene: inherit randomly. */
                const NEAT_ConnGene *src = (rng_f() < 0.5) ? fc : &other->conns[j];
                child->conns[child->n_conns] = *src;
                /* If either parent disabled, 75% chance offspring also disabled. */
                if (!fc->enabled || !other->conns[j].enabled)
                    child->conns[child->n_conns].enabled = (rng_f() < 0.25) ? 1 : 0;
                else
                    child->conns[child->n_conns].enabled = 1;
                ++child->n_conns;
                found = 1;
                break;
            }
        }
        if (!found)
            child->conns[child->n_conns++] = *fc;
    }

    /* Ensure all nodes referenced by inherited connections exist in child. */
    for (int ci = 0; ci < child->n_conns; ++ci) {
        int ids[2] = { child->conns[ci].in_node, child->conns[ci].out_node };
        for (int k = 0; k < 2; ++k) {
            if (genome_find_node(child, ids[k]) >= 0) continue;
            /* Try to find in either parent. */
            int ni = genome_find_node(fitter, ids[k]);
            if (ni < 0) ni = genome_find_node(other, ids[k]);
            if (ni >= 0) {
                /* ni references either fitter or other; search both explicitly. */
                NEAT_NodeGene ng;
                int nfi = genome_find_node(fitter, ids[k]);
                if (nfi >= 0) ng = fitter->nodes[nfi];
                else {
                    int noi = genome_find_node(other, ids[k]);
                    ng = other->nodes[noi];
                }
                child->nodes = realloc(child->nodes,
                    (child->n_nodes + 1) * sizeof(NEAT_NodeGene));
                child->nodes[child->n_nodes++] = ng;
            }
        }
    }

    return child;
}

/* =========================================================================
 * Network decode & activation
 * ========================================================================= */

NEAT_Network *neat_decode(const NEAT_Genome *g, int n_inputs, int n_outputs) {
    NEAT_Network *net = calloc(1, sizeof(NEAT_Network));

    net->n_nodes     = g->n_nodes;
    net->n_inputs    = n_inputs;
    net->n_outputs   = n_outputs;
    net->n_activations = 1;

    net->node_ids    = malloc(g->n_nodes * sizeof(int));
    net->activations = calloc(g->n_nodes, sizeof(double));

    for (int i = 0; i < g->n_nodes; ++i)
        net->node_ids[i] = g->nodes[i].id;

    int nc = 0;
    for (int i = 0; i < g->n_conns; ++i)
        if (g->conns[i].enabled) ++nc;

    net->n_conns = nc;
    net->conn_in = malloc((nc ? nc : 1) * sizeof(int));
    net->conn_out= malloc((nc ? nc : 1) * sizeof(int));
    net->conn_w  = malloc((nc ? nc : 1) * sizeof(double));

    int ci = 0;
    for (int i = 0; i < g->n_conns; ++i) {
        if (!g->conns[i].enabled) continue;
        int in_idx = -1, out_idx = -1;
        for (int j = 0; j < g->n_nodes; ++j) {
            if (g->nodes[j].id == g->conns[i].in_node)  in_idx  = j;
            if (g->nodes[j].id == g->conns[i].out_node) out_idx = j;
        }
        if (in_idx < 0 || out_idx < 0) continue;
        net->conn_in [ci] = in_idx;
        net->conn_out[ci] = out_idx;
        net->conn_w  [ci] = g->conns[i].weight;
        ++ci;
    }
    net->n_conns = ci;

    /* Count each node type for index arrays. */
    int n_bias = 0;
    for (int i = 0; i < g->n_nodes; ++i)
        if (g->nodes[i].type == NEAT_NODE_BIAS) ++n_bias;

    net->n_bias     = n_bias;
    net->input_idx  = malloc((n_inputs  ? n_inputs  : 1) * sizeof(int));
    net->output_idx = malloc((n_outputs ? n_outputs : 1) * sizeof(int));
    net->bias_idx   = malloc((n_bias    ? n_bias    : 1) * sizeof(int));

    int ii = 0, oi = 0, bi = 0;
    for (int i = 0; i < g->n_nodes; ++i) {
        int t = g->nodes[i].type;
        if      (t == NEAT_NODE_SENSOR && ii < n_inputs)  net->input_idx[ii++]  = i;
        else if (t == NEAT_NODE_OUTPUT && oi < n_outputs) net->output_idx[oi++] = i;
        else if (t == NEAT_NODE_BIAS)                     net->bias_idx[bi++]   = i;
    }

    /* Initialise bias activations to 1.0. */
    for (int i = 0; i < n_bias; ++i)
        net->activations[net->bias_idx[i]] = 1.0;

    return net;
}

/* Modified sigmoid: phi(x) = 1 / (1 + exp(-4.9*x))  (Section 4.1). */
static double sigmoid(double x) { return 1.0 / (1.0 + exp(-4.9 * x)); }

void neat_reset(NEAT_Network *net) {
    memset(net->activations, 0, net->n_nodes * sizeof(double));
    for (int i = 0; i < net->n_bias; ++i)
        net->activations[net->bias_idx[i]] = 1.0;
}

/*
 * Activate a feed-forward network in topological order.
 *
 * For recurrent or large networks, n_activations > 1 uses iterative update.
 * For standard feed-forward use, one topological pass is correct and fast.
 *
 * Topological sort: Kahn's algorithm on the connection graph.
 * Sensor and bias nodes are sources (in-degree 0 by definition).
 * Non-fixed nodes are sorted so every source precedes its targets.
 * This ensures hidden nodes are evaluated before the output nodes that
 * depend on them — critical for multi-layer topologies.
 */
void neat_activate(NEAT_Network *net, const double *inputs, double *outputs) {
    const int N = net->n_nodes;

    /* Mark fixed (sensor + bias) nodes. */
    char *fixed = calloc(N, 1);
    for (int i = 0; i < net->n_inputs; ++i) {
        net->activations[net->input_idx[i]] = inputs[i];
        fixed[net->input_idx[i]] = 1;
    }
    for (int i = 0; i < net->n_bias; ++i) {
        net->activations[net->bias_idx[i]] = 1.0;
        fixed[net->bias_idx[i]] = 1;
    }

    if (net->n_activations <= 1) {
        /*
         * Single topological-order pass (correct for feed-forward networks).
         *
         * Kahn's algorithm: compute in-degree counting only edges whose source
         * is a non-fixed node (fixed nodes are already evaluated).  Process
         * nodes one by one; apply sigmoid immediately so downstream nodes see
         * the updated value when they are later dequeued.
         */
        int *indeg = calloc(N, sizeof(int));
        for (int c = 0; c < net->n_conns; ++c)
            if (!fixed[net->conn_in[c]])
                indeg[net->conn_out[c]]++;

        double *sum = calloc(N, sizeof(double));

        /* Seed sums from fixed nodes and enqueue nodes whose indeg becomes 0. */
        int *queue = malloc(N * sizeof(int));
        int head = 0, tail = 0;
        /* First pass: accumulate contributions from fixed sources. */
        for (int c = 0; c < net->n_conns; ++c) {
            if (fixed[net->conn_in[c]]) {
                int v = net->conn_out[c];
                sum[v] += net->activations[net->conn_in[c]] * net->conn_w[c];
            }
        }
        /* Enqueue non-fixed nodes whose in-degree is 0 (no non-fixed predecessors). */
        for (int i = 0; i < N; ++i)
            if (!fixed[i] && indeg[i] == 0)
                queue[tail++] = i;

        while (head < tail) {
            int u = queue[head++];
            /* Evaluate this node. */
            net->activations[u] = sigmoid(sum[u]);
            /* Propagate to successors. */
            for (int c = 0; c < net->n_conns; ++c) {
                if (net->conn_in[c] != u) continue;
                int v = net->conn_out[c];
                sum[v] += net->activations[u] * net->conn_w[c];
                if (--indeg[v] == 0)
                    queue[tail++] = v;
            }
        }
        free(sum);
        free(queue);
        free(indeg);
    } else {
        /*
         * Iterative update for recurrent networks (n_activations passes).
         * Each pass: sum all incoming weighted activations, apply sigmoid.
         * Sensor/bias values are re-pinned after every iteration.
         */
        for (int iter = 0; iter < net->n_activations; ++iter) {
            double *sum = calloc(N, sizeof(double));
            for (int c = 0; c < net->n_conns; ++c)
                sum[net->conn_out[c]] += net->activations[net->conn_in[c]] * net->conn_w[c];
            for (int i = 0; i < N; ++i) {
                if (fixed[i]) continue;
                net->activations[i] = sigmoid(sum[i]);
            }
            free(sum);
            /* Re-pin sensor/bias after sigmoid pass. */
            for (int i = 0; i < net->n_inputs; ++i)
                net->activations[net->input_idx[i]] = inputs[i];
            for (int i = 0; i < net->n_bias; ++i)
                net->activations[net->bias_idx[i]] = 1.0;
        }
    }

    free(fixed);
    for (int i = 0; i < net->n_outputs; ++i)
        outputs[i] = net->activations[net->output_idx[i]];
}

void neat_network_free(NEAT_Network *net) {
    if (!net) return;
    free(net->node_ids);
    free(net->activations);
    free(net->conn_in);
    free(net->conn_out);
    free(net->conn_w);
    free(net->input_idx);
    free(net->output_idx);
    free(net->bias_idx);
    free(net);
}

void neat_result_free(NEAT_Result *r) {
    if (!r) return;
    neat_genome_free(r->best);
    r->best = NULL;
}

/* =========================================================================
 * Species
 * ========================================================================= */

typedef struct {
    int     id;
    int    *members;
    int     n_members;
    int     cap;
    int     rep_idx;
    double  best_fitness;
    int     stagnation;
    int     offspring;
} Species;

static void species_free(Species *s) { free(s->members); }

static void species_add(Species *s, int idx) {
    if (s->n_members == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 8;
        s->members = realloc(s->members, s->cap * sizeof(int));
    }
    s->members[s->n_members++] = idx;
}

/* =========================================================================
 * Main NEAT loop
 * ========================================================================= */

/* Create minimal initial genome: inputs+bias fully connected to outputs. */
static NEAT_Genome *make_minimal_genome(int n_inputs, int n_outputs,
                                        int add_bias, InnovDB *db,
                                        double weight_range) {
    NEAT_Genome *g = genome_alloc(0, 0);

    for (int i = 0; i < n_inputs; ++i)
        genome_add_node(g, i + 1, NEAT_NODE_SENSOR);

    int bias_id = 0;
    if (add_bias) {
        bias_id = n_inputs + 1;
        genome_add_node(g, bias_id, NEAT_NODE_BIAS);
    }

    int out_start = n_inputs + (add_bias ? 1 : 0) + 1;
    for (int i = 0; i < n_outputs; ++i)
        genome_add_node(g, out_start + i, NEAT_NODE_OUTPUT);

    for (int i = 0; i < n_inputs; ++i) {
        for (int o = 0; o < n_outputs; ++o) {
            int innov = innovdb_get(db, i + 1, out_start + o);
            genome_add_conn(g, i + 1, out_start + o, rng_sym(weight_range), 1, innov);
        }
    }
    if (add_bias) {
        for (int o = 0; o < n_outputs; ++o) {
            int innov = innovdb_get(db, bias_id, out_start + o);
            genome_add_conn(g, bias_id, out_start + o, rng_sym(weight_range), 1, innov);
        }
    }
    return g;
}

/* Apply defaults to zero fields in NEAT_Params. */
static void apply_defaults(NEAT_Params *p) {
    if (p->pop_size          == 0)   p->pop_size          = 150;
    if (p->max_gens          == 0)   p->max_gens          = 500;
    if (p->target_fitness    == 0.0) p->target_fitness    = DBL_MAX;
    if (p->c1                == 0.0) p->c1                = 1.0;
    if (p->c2                == 0.0) p->c2                = 1.0;
    if (p->c3                == 0.0) p->c3                = 0.4;
    if (p->delta_t           == 0.0) p->delta_t           = 3.0;
    if (p->p_weight_mut      == 0.0) p->p_weight_mut      = 0.80;
    if (p->p_weight_perturb  == 0.0) p->p_weight_perturb  = 0.90;
    if (p->weight_perturb    == 0.0) p->weight_perturb    = 0.5;   /* wider perturbation */
    if (p->weight_range      == 0.0) p->weight_range      = 2.0;   /* ±2 initial weights */
    if (p->p_add_conn        == 0.0) p->p_add_conn        = 0.05;
    if (p->p_add_node        == 0.0) p->p_add_node        = 0.03;
    if (p->p_no_crossover    == 0.0) p->p_no_crossover    = 0.25;
    if (p->interspecies_rate == 0.0) p->interspecies_rate = 0.001;
    if (p->survival_rate     == 0.0) p->survival_rate     = 0.20;
    if (p->stagnation_limit  == 0)   p->stagnation_limit  = 15;
    if (p->add_bias == 0) p->add_bias = 1;
}

NEAT_Result neat_run(NEAT_Params *p) {
    apply_defaults(p);
    rng_seed(p->seed);

    const int N  = p->pop_size;
    int total_evals = 0;

    InnovDB db;
    innovdb_init(&db, 1);

    /* Initialise population (Section 3.4: minimal topologies, no hidden nodes). */
    NEAT_Genome **pop = malloc(N * sizeof(NEAT_Genome *));
    for (int i = 0; i < N; ++i)
        pop[i] = make_minimal_genome(p->n_inputs, p->n_outputs,
                                     p->add_bias, &db, p->weight_range);

    int       n_species = 0, species_cap = 16, next_species_id = 1;
    Species  *specs = calloc(species_cap, sizeof(Species));

    NEAT_Genome *best_genome = NULL;
    double       best_fitness = -DBL_MAX;
    int          stop_flag = NEAT_STOP_MAX_GENS;
    int          gen;

    for (gen = 0; gen < p->max_gens; ++gen) {
        /* ----------------------------------------------------------------
         * Step 1: Evaluate fitness.
         * ---------------------------------------------------------------- */
        innovdb_new_gen(&db);

        for (int i = 0; i < N; ++i) {
            NEAT_Network *net = neat_decode(pop[i], p->n_inputs, p->n_outputs);
            pop[i]->fitness = p->fitness_fn(net, p->userdata);
            neat_network_free(net);
            ++total_evals;

            if (pop[i]->fitness > best_fitness) {
                best_fitness = pop[i]->fitness;
                neat_genome_free(best_genome);
                best_genome = genome_copy(pop[i]);
            }
            if (pop[i]->fitness >= p->target_fitness) {
                stop_flag = NEAT_STOP_TARGET;
                goto done;
            }
        }

        /* ----------------------------------------------------------------
         * Step 2: Speciation (Section 3.3).
         * ---------------------------------------------------------------- */
        for (int s = 0; s < n_species; ++s)
            specs[s].n_members = 0;

        for (int i = 0; i < N; ++i) {
            int placed = 0;
            for (int s = 0; s < n_species; ++s) {
                NEAT_Genome *rep = pop[specs[s].rep_idx];
                double d = neat_compat(pop[i], rep, p->c1, p->c2, p->c3);
                if (d < p->delta_t) {
                    species_add(&specs[s], i);
                    pop[i]->species_id = specs[s].id;
                    placed = 1;
                    break;
                }
            }
            if (!placed) {
                if (n_species == species_cap) {
                    species_cap *= 2;
                    specs = realloc(specs, species_cap * sizeof(Species));
                    memset(&specs[n_species], 0,
                           (species_cap/2) * sizeof(Species));
                }
                Species *ns       = &specs[n_species];
                ns->id            = next_species_id++;
                ns->n_members     = 0;
                ns->cap           = 0;
                ns->members       = NULL;
                ns->rep_idx       = i;
                ns->best_fitness  = -DBL_MAX;
                ns->stagnation    = 0;
                ns->offspring     = 0;
                species_add(ns, i);
                pop[i]->species_id = ns->id;
                ++n_species;
            }
        }

        /* Remove empty species. */
        for (int s = 0; s < n_species; ) {
            if (specs[s].n_members == 0) {
                species_free(&specs[s]);
                specs[s] = specs[--n_species];
                memset(&specs[n_species], 0, sizeof(Species));
            } else {
                ++s;
            }
        }

        /* Update stagnation and choose a random representative. */
        for (int s = 0; s < n_species; ++s) {
            double sbest = -DBL_MAX;
            for (int m = 0; m < specs[s].n_members; ++m) {
                double f = pop[specs[s].members[m]]->fitness;
                if (f > sbest) sbest = f;
            }
            if (sbest > specs[s].best_fitness) {
                specs[s].best_fitness = sbest;
                specs[s].stagnation   = 0;
            } else {
                ++specs[s].stagnation;
            }
            specs[s].rep_idx = specs[s].members[rng_int(specs[s].n_members)];
        }

        /* ----------------------------------------------------------------
         * Step 3: Fitness sharing (Eq. 2) and offspring allocation.
         * Adjusted fitness of species s = mean(f_i) / n_members (explicit sharing).
         * Offspring ∝ adjusted_fitness, normalised to sum to N.
         * ---------------------------------------------------------------- */
        double total_adj = 0.0;
        double *adj = calloc(n_species, sizeof(double));
        for (int s = 0; s < n_species; ++s) {
            if (specs[s].stagnation >= p->stagnation_limit && n_species > 2) {
                adj[s] = 0.0;
                specs[s].offspring = 0;
                continue;
            }
            double sum = 0.0;
            for (int m = 0; m < specs[s].n_members; ++m)
                sum += pop[specs[s].members[m]]->fitness;
            /* Shift by minimum fitness if any genome has negative fitness. */
            adj[s] = sum / specs[s].n_members;
            if (adj[s] < 0.0) adj[s] = 0.0;
            total_adj += adj[s];
        }
        if (total_adj <= 0.0) {
            /* All stagnant or zero fitness: give equal share. */
            for (int s = 0; s < n_species; ++s)
                adj[s] = 1.0;
            total_adj = n_species;
        }
        /* Allocate proportionally; fix rounding by adjusting largest species. */
        int assigned = 0;
        int best_s = 0;
        for (int s = 0; s < n_species; ++s) {
            specs[s].offspring = (int)round(adj[s] / total_adj * N);
            if (specs[s].offspring < 0) specs[s].offspring = 0;
            assigned += specs[s].offspring;
            if (specs[s].offspring > specs[best_s].offspring) best_s = s;
        }
        specs[best_s].offspring += (N - assigned);
        if (specs[best_s].offspring < 0) specs[best_s].offspring = 0;
        free(adj);

        /* ----------------------------------------------------------------
         * Step 4: Reproduce — build new population.
         * ---------------------------------------------------------------- */
        NEAT_Genome **new_pop = malloc(N * sizeof(NEAT_Genome *));
        int new_idx = 0;

        for (int s = 0; s < n_species && new_idx < N; ++s) {
            if (specs[s].offspring <= 0) continue;

            int nm = specs[s].n_members;
            /* Sort descending by fitness. */
            for (int a = 0; a < nm - 1; ++a)
                for (int b = a + 1; b < nm; ++b)
                    if (pop[specs[s].members[b]]->fitness >
                        pop[specs[s].members[a]]->fitness) {
                        int tmp = specs[s].members[a];
                        specs[s].members[a] = specs[s].members[b];
                        specs[s].members[b] = tmp;
                    }

            /* Elitism: copy champion unchanged. */
            new_pop[new_idx++] = genome_copy(pop[specs[s].members[0]]);
            int made = 1;

            int survivors = (int)(nm * p->survival_rate);
            if (survivors < 1) survivors = 1;
            if (survivors > nm) survivors = nm;

            while (made < specs[s].offspring && new_idx < N) {
                NEAT_Genome *child;
                if (rng_f() < p->p_no_crossover || survivors == 1) {
                    int pi = specs[s].members[rng_int(survivors)];
                    child = genome_copy(pop[pi]);
                } else {
                    int ai = specs[s].members[rng_int(survivors)];
                    int bi;
                    if (rng_f() < p->interspecies_rate && n_species > 1) {
                        int other_s = rng_int(n_species - 1);
                        if (other_s >= s) ++other_s;
                        int om = specs[other_s].n_members;
                        bi = specs[other_s].members[rng_int(om)];
                    } else {
                        bi = specs[s].members[rng_int(survivors)];
                    }
                    const NEAT_Genome *fitter = (pop[ai]->fitness >= pop[bi]->fitness)
                                                ? pop[ai] : pop[bi];
                    const NEAT_Genome *other  = (fitter == pop[ai]) ? pop[bi] : pop[ai];
                    child = crossover(fitter, other);
                }

                if (rng_f() < p->p_weight_mut)
                    mutate_weights(child, p->p_weight_perturb,
                                   p->weight_perturb, p->weight_range);
                if (rng_f() < p->p_add_node)
                    mutate_add_node(child, &db);
                if (rng_f() < p->p_add_conn)
                    mutate_add_conn(child, &db, p->weight_range);

                new_pop[new_idx++] = child;
                ++made;
            }
        }

        /* Fill any remaining slots due to rounding (copy from best species). */
        while (new_idx < N) {
            int best_s2 = 0;
            for (int s = 1; s < n_species; ++s)
                if (specs[s].n_members > specs[best_s2].n_members) best_s2 = s;
            int pi = specs[best_s2].members[rng_int(specs[best_s2].n_members)];
            NEAT_Genome *child = genome_copy(pop[pi]);
            if (rng_f() < p->p_weight_mut)
                mutate_weights(child, p->p_weight_perturb,
                               p->weight_perturb, p->weight_range);
            new_pop[new_idx++] = child;
        }

        for (int i = 0; i < N; ++i) neat_genome_free(pop[i]);
        free(pop);
        pop = new_pop;
    }

done:
    for (int i = 0; i < N; ++i) neat_genome_free(pop[i]);
    free(pop);

    for (int s = 0; s < n_species; ++s) species_free(&specs[s]);
    free(specs);
    innovdb_free(&db);

    NEAT_Result r;
    r.best         = best_genome;
    r.best_fitness = best_fitness;
    r.gens         = gen;
    r.evals        = total_evals;
    r.n_species    = n_species;
    r.stop_flag    = stop_flag;
    return r;
}
