#ifndef DM_VOLT_H
#define DM_VOLT_H

/*
 * VOLT: Vocabulary Learning via Optimal Transport
 * ACL 2021 - Xu, Zhou, Gan, Zheng, Li
 *
 * Algorithm 1 (VOLT):
 *   Input:  BPE token candidates L ranked by frequency,
 *           incremental size sequence S, character set C, corpus Dc
 *   Output: vocabulary v* with maximum Marginal Utility of Vocabularization
 *
 * Marginal Utility (Eq.1): MUV(v(t)) = -(H_v(t) - H_v(t-1)) / (S[t]-S[t-1])
 * Entropy (Eq.2):          H_v = -1/l_v * sum_{i in v} P(i)*log P(i)
 * Kernel:                  K[i][j] = 1/len(T[i]) if char_j in T[i], else 0
 * Sinkhorn:                u = P(T)/(K*v), v = P(C)/(K^T*u), P_opt = diag(u)*K*diag(v)
 */

int dm_volt_cli(int argc, char **argv);

#endif /* DM_VOLT_H */
