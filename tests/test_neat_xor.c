/*
 * NEAT smoke test: XOR (Section 4.2 of the paper).
 * XOR cannot be solved without hidden nodes; NEAT should grow structure.
 */
#include "algorithms/neat.h"
#include <stdio.h>
#include <math.h>

static const double XOR_IN[4][2] = {{0,0},{0,1},{1,0},{1,1}};
static const double XOR_OUT[4]   = {0.0, 1.0, 1.0, 0.0};

static double xor_fitness(NEAT_Network *net, void *ud) {
    (void)ud;
    double err = 0.0;
    double out[1];
    for (int i = 0; i < 4; ++i) {
        neat_reset(net);
        neat_activate(net, XOR_IN[i], out);
        double d = XOR_OUT[i] - out[0];
        err += d * d;
    }
    /* fitness = (4 - sum_of_squared_errors)^2  (paper Section 4.2) */
    double f = 4.0 - sqrt(err);
    return f * f;
}

int main(void) {
    printf("NEAT XOR test (paper Section 4.2)\n");

    NEAT_Params p;
    memset(&p, 0, sizeof(p));
    p.n_inputs       = 2;
    p.n_outputs      = 1;
    p.pop_size       = 150;
    p.max_gens       = 200;
    p.target_fitness = 15.9;
    p.fitness_fn     = xor_fitness;
    p.seed           = 42;

    NEAT_Result r = neat_run(&p);

    printf("  Generations: %d\n", r.gens);
    printf("  Evaluations: %d\n", r.evals);
    printf("  Best fitness: %.4f\n", r.best_fitness);
    printf("  Species at end: %d\n", r.n_species);
    printf("  Stop: %s\n", r.stop_flag == NEAT_STOP_TARGET ? "TARGET" : "MAX_GENS");

    if (r.best) {
        printf("  Best genome: %d nodes, %d conns\n",
               r.best->n_nodes, r.best->n_conns);
        int hidden = 0;
        for (int i = 0; i < r.best->n_nodes; ++i)
            if (r.best->nodes[i].type == NEAT_NODE_HIDDEN) ++hidden;
        printf("  Hidden nodes: %d\n", hidden);

        NEAT_Network *net = neat_decode(r.best, p.n_inputs, p.n_outputs);
        printf("  XOR outputs:\n");
        int solved = 1;
        double out[1];
        for (int i = 0; i < 4; ++i) {
            neat_reset(net);
            neat_activate(net, XOR_IN[i], out);
            int pred = out[0] > 0.5 ? 1 : 0;
            int expected = (int)XOR_OUT[i];
            if (pred != expected) solved = 0;
            printf("    [%g,%g] -> %.4f (%s)\n",
                   XOR_IN[i][0], XOR_IN[i][1], out[0],
                   pred == expected ? "OK" : "WRONG");
        }
        neat_network_free(net);
        printf("  XOR %s\n", solved ? "SOLVED" : "not solved in this run");
    }

    neat_result_free(&r);
    return 0;
}
