/*
 * NEAT: NeuroEvolution of Augmenting Topologies.
 *
 * Stanley, K. O., and Miikkulainen, R., "Evolving Neural Networks through
 * Augmenting Topologies," Evolutionary Computation, MIT Press, 10(2):99-127,
 * 2002. Technical Report TR-AI-01-290, UT Austin, 2001.
 *
 * Three key innovations (Section 2):
 *  (1) Historical markings: global innovation numbers track gene homology,
 *      enabling crossover across topologies of different sizes (Section 3.2).
 *  (2) Speciation: explicit fitness sharing (Eq. 2) groups similar genomes
 *      into species, protecting structural innovation (Section 3.3).
 *  (3) Minimal initial structure: population starts with no hidden nodes,
 *      growing complexity only as beneficial (Section 3.4).
 *
 * Genetic encoding (Section 3.1, Figure 2):
 *  - NodeGene: node id, type (sensor/hidden/output/bias).
 *  - ConnGene: in_node, out_node, weight, enabled, innovation_number.
 *  Structural mutations (Figure 3):
 *  - Add connection: new conn gene with next global innovation number.
 *  - Add node: split existing conn; disable original; add two new conns
 *    (weight-1 into new node, old weight out of new node).
 *
 * Crossover (Section 3.2, Figure 4):
 *  Align genomes by innovation number. Matching genes: inherited randomly.
 *  Disjoint/excess genes: always from the more fit parent (or randomly if
 *  equal fitness).
 *
 * Compatibility distance (Eq. 1):
 *  delta = c1*E/N + c2*D/N + c3*W_bar
 *  E=excess genes, D=disjoint genes, W_bar=average weight diff of matching
 *  genes (including disabled), N=genes in larger genome.
 *
 * Fitness sharing (Eq. 2):
 *  f'_i = f_i / sum_j sh(delta(i,j))
 *  sh(d) = 1 if d < delta_t, else 0.
 *
 * Activation: modified sigmoid phi(x) = 1 / (1 + exp(-4.9*x)) (Section 4.1).
 *
 * Default hyperparameters (Section 4.1):
 *  population=150, c1=c2=1.0, c3=0.4, delta_t=3.0,
 *  p_weight_mut=0.80, p_weight_perturb=0.90, p_add_conn=0.05,
 *  p_add_node=0.03, p_no_crossover=0.25, interspecies_rate=0.001,
 *  stagnation_limit=15 generations.
 *
 * Usage:
 *   NEAT_Params p; memset(&p, 0, sizeof(p));
 *   p.n_inputs  = 3;
 *   p.n_outputs = 1;
 *   p.pop_size  = 150;
 *   p.max_gens  = 300;
 *   p.fitness_fn = my_eval;
 *   p.target_fitness = 3.9;
 *   NEAT_Result r = neat_run(&p);
 *   // activate the best network:
 *   double out[1];
 *   neat_activate(r.best, inputs, out, p.n_inputs, p.n_outputs);
 *   neat_result_free(&r);
 */

#ifndef DM_NEAT_H
#define DM_NEAT_H

#include "core/dm_common.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Constants
 * ------------------------------------------------------------------------- */

#define NEAT_NODE_SENSOR  0
#define NEAT_NODE_HIDDEN  1
#define NEAT_NODE_OUTPUT  2
#define NEAT_NODE_BIAS    3

/* -------------------------------------------------------------------------
 * Genome data structures
 * ------------------------------------------------------------------------- */

/* A single node gene. */
typedef struct {
    int  id;    /* unique node id (1-based) */
    int  type;  /* NEAT_NODE_* */
} NEAT_NodeGene;

/* A single connection gene. */
typedef struct {
    int    in_node;    /* source node id */
    int    out_node;   /* target node id */
    double weight;
    int    enabled;    /* 1=expressed, 0=disabled */
    int    innov;      /* global innovation number */
} NEAT_ConnGene;

/*
 * A genome = a genotype representing one neural network.
 * Encodes both topology and weights (Figure 2).
 */
typedef struct {
    NEAT_NodeGene *nodes;   /* node gene array */
    int            n_nodes;
    NEAT_ConnGene *conns;   /* connection gene array, sorted by innov */
    int            n_conns;
    double         fitness;
    int            species_id;
} NEAT_Genome;

/*
 * A phenotype network decoded from a genome for activation.
 * Neurons are evaluated in topological order (feed-forward) or via
 * iteration (recurrent); n_activations iterations are used for recurrent.
 */
typedef struct {
    int     n_nodes;
    int    *node_ids;          /* [n_nodes] node id -> index mapping */
    double *activations;       /* [n_nodes] current activation values */
    /* Connections: parallel arrays */
    int     n_conns;
    int    *conn_in;           /* [n_conns] index into activations */
    int    *conn_out;          /* [n_conns] index into activations */
    double *conn_w;            /* [n_conns] weights */
    /* Input/output/bias node indices into activations[] */
    int    *input_idx;         /* [n_inputs] */
    int    *output_idx;        /* [n_outputs] */
    int    *bias_idx;          /* [n_bias] */
    int     n_inputs;
    int     n_outputs;
    int     n_bias;
    int     n_activations;     /* recurrent update iterations (default 1) */
} NEAT_Network;

/* -------------------------------------------------------------------------
 * Fitness function callback
 * -------------------------------------------------------------------------
 * Evaluate a single network. Returns fitness (higher=better).
 * net: decoded phenotype ready for neat_activate().
 * userdata: passed through unchanged.
 */
typedef double (*NEAT_FitnessFn)(NEAT_Network *net, void *userdata);

/* -------------------------------------------------------------------------
 * Hyperparameters (Section 4.1 defaults applied by neat_run if zero)
 * ------------------------------------------------------------------------- */

typedef struct {
    /* Problem definition */
    int    n_inputs;         /* number of sensor nodes (required) */
    int    n_outputs;        /* number of output nodes (required) */
    int    add_bias;         /* 1 = add a bias node (default 1) */

    /* Population */
    int    pop_size;         /* population size (default 150) */
    int    max_gens;         /* generation limit (default 500) */

    /* Termination */
    double target_fitness;   /* stop when any genome reaches this (default HUGE_VAL) */

    /* Compatibility distance (Eq. 1) */
    double c1;               /* excess coefficient (default 1.0) */
    double c2;               /* disjoint coefficient (default 1.0) */
    double c3;               /* weight diff coefficient (default 0.4) */
    double delta_t;          /* compatibility threshold (default 3.0) */

    /* Mutation rates */
    double p_weight_mut;     /* prob genome has weights mutated (default 0.80) */
    double p_weight_perturb; /* prob each weight is perturbed vs reset (default 0.90) */
    double weight_perturb;   /* perturb range [-w, w] (default 0.1) */
    double weight_range;     /* reset range (default 1.0) */
    double p_add_conn;       /* prob structural mutation: add connection (default 0.05) */
    double p_add_node;       /* prob structural mutation: add node (default 0.03) */

    /* Reproduction */
    double p_no_crossover;   /* prob offspring from mutation only (default 0.25) */
    double interspecies_rate;/* prob of interspecies mating (default 0.001) */
    double survival_rate;    /* fraction of each species kept for reproduction (default 0.2) */

    /* Speciation */
    int    stagnation_limit; /* gens without improvement before species culled (default 15) */

    /* Evaluation */
    NEAT_FitnessFn fitness_fn;
    void          *userdata;

    /* RNG seed (0 = time-based) */
    unsigned long  seed;
} NEAT_Params;

/* -------------------------------------------------------------------------
 * Result
 * ------------------------------------------------------------------------- */

typedef struct {
    NEAT_Genome *best;       /* best genome found (heap-alloc; free with neat_genome_free) */
    double       best_fitness;
    int          gens;       /* generations run */
    int          evals;      /* fitness evaluations performed */
    int          n_species;  /* species count in final generation */
    int          stop_flag;  /* reason for stopping */
} NEAT_Result;

#define NEAT_STOP_MAX_GENS   1
#define NEAT_STOP_TARGET     2

/* -------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */

/*
 * Run NEAT and return the best genome found.
 * Fills default hyperparameter values for any zero fields in p.
 */
NEAT_Result neat_run(NEAT_Params *p);

/*
 * Decode a genome into a network ready for activation.
 * Caller must free with neat_network_free().
 */
NEAT_Network *neat_decode(const NEAT_Genome *g, int n_inputs, int n_outputs);

/*
 * Activate a decoded network.
 *   inputs  : [n_inputs]  — sensor values
 *   outputs : [n_outputs] — written with network outputs
 * The modified sigmoid phi(x) = 1/(1+exp(-4.9*x)) is used (Section 4.1).
 */
void neat_activate(NEAT_Network *net, const double *inputs, double *outputs);

/* Reset all hidden/output activations to 0 (for recurrent networks). */
void neat_reset(NEAT_Network *net);

/* Memory management */
void neat_genome_free(NEAT_Genome *g);
void neat_network_free(NEAT_Network *net);
void neat_result_free(NEAT_Result *r);

/* Compatibility distance between two genomes (Eq. 1). */
double neat_compat(const NEAT_Genome *a, const NEAT_Genome *b,
                   double c1, double c2, double c3);

#ifdef __cplusplus
}
#endif

#endif /* DM_NEAT_H */
