#include "attacks.h"

// index: square
U64 knight_attacks[64];

void init_knight_attacks() {
    for (int sq=0; sq<64; sq++) {
        knight_attacks[sq] = 0ULL;
        U64 sq_bitmask = sq_bit[sq];

        if (sq_bitmask & not_rank_8) {
            if (sq_bitmask & not_ab_file) { knight_attacks[sq] |= sq_bitmask >> 10; }
            if (sq_bitmask & not_gh_file) { knight_attacks[sq] |= sq_bitmask >> 6; }
            if (sq_bitmask & not_rank_7) {
                if (sq_bitmask & not_a_file) { knight_attacks[sq] |= sq_bitmask >> 17; }
                if (sq_bitmask & not_h_file) { knight_attacks[sq] |= sq_bitmask >> 15; }
            }
        }

        if (sq_bitmask & not_rank_1) {
            if (sq_bitmask & not_ab_file) { knight_attacks[sq] |= sq_bitmask << 6; }
            if (sq_bitmask & not_gh_file) { knight_attacks[sq] |= sq_bitmask << 10; }
            if (sq_bitmask & not_rank_2) {
                if (sq_bitmask & not_a_file) { knight_attacks[sq] |= sq_bitmask << 15; }
                if (sq_bitmask & not_h_file) { knight_attacks[sq] |= sq_bitmask << 17; }
            }
        }
    }
}





