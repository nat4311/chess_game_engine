#include "attacks.h"

// index: square
U64 king_attacks[64];

void init_king_attacks() {
    for (int sq=0; sq<64; sq++) {
        king_attacks[sq] = 0ULL;
        U64 sq_bitmask = sq_bit[sq];

        if (sq_bitmask & not_rank_8) {
            king_attacks[sq] |= sq_bitmask >> 8;
            if (sq_bitmask & not_a_file) { king_attacks[sq] |= sq_bitmask >> 9; }
            if (sq_bitmask & not_h_file) { king_attacks[sq] |= sq_bitmask >> 7; }
        }

        if (sq_bitmask & not_a_file) { king_attacks[sq] |= sq_bitmask >> 1; }
        if (sq_bitmask & not_h_file) { king_attacks[sq] |= sq_bitmask << 1; }

        if (sq_bitmask & not_rank_1) {
            king_attacks[sq] |= sq_bitmask << 8;
            if (sq_bitmask & not_a_file) { king_attacks[sq] |= sq_bitmask << 7; }
            if (sq_bitmask & not_h_file) { king_attacks[sq] |= sq_bitmask << 9; }
        }
    }
}
