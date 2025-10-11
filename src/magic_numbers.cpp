/* references:
 * https://www.chessprogramming.org/Looking_for_Magics
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "engine.h"
#include "attacks.cpp"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>

/*/////////////////////////////////////////////////////////////////////////////
                                helper functions
/*/////////////////////////////////////////////////////////////////////////////

#define lsb_scan(x) __builtin_ctz(x)
#define bit_count(x) __builtin_popcountll(x)


unsigned int random_state = 1804289383;

unsigned int get_random_U32_number() { // XOR shift algorithm
    unsigned int n = random_state;

    n ^= n << 13;
    n ^= n >> 17;
    n ^= n << 5;
    random_state = n;

    return random_state;
}

U64 random_U64() {
    U64 a, b, c , d;
    a = (U64)(get_random_U32_number() & 0xffff);
    b = (U64)(get_random_U32_number() & 0xffff);
    c = (U64)(get_random_U32_number() & 0xffff);
    d = (U64)(get_random_U32_number() & 0xffff);
    return a | (b<<16) | (c<<32) | (d<<48);
    // return a | (b<<32);
}
U64 random_U64_few_bits() {
    return random_U64() & random_U64() & random_U64();
}

/*/////////////////////////////////////////////////////////////////////////////
                                   bishop
/*/////////////////////////////////////////////////////////////////////////////

int bishop_n_relevant_occupancies[64] = {
    6, 5, 5, 5, 5, 5, 5, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 5, 5, 5, 5, 5, 5, 6,
};

// indices: square, occupancy_index
U64 bishop_relevant_occupancies[64][512];

U64 bishop_relevant_occupancy(int square, int occupancy_index) {
    U64 relevant_occupancy = 0;
    int sqs[9];
    int sqs_index = 0;
    int x0 = square % 8;
    int y0 = square / 8;
    int x, y;

    for (x=x0-1,y=y0-1; x>0 && y>0; x--,y--) { sqs[sqs_index++] = x+y*8; }
    for (x=x0-1,y=y0+1; x>0 && y<7; x--,y++) { sqs[sqs_index++] = x+y*8; }
    for (x=x0+1,y=y0-1; x<7 && y>0; x++,y--) { sqs[sqs_index++] = x+y*8; }
    for (x=x0+1,y=y0+1; x<7 && y<7; x++,y++) { sqs[sqs_index++] = x+y*8; }

    sqs_index = 0;
    while (occupancy_index) {
        if (occupancy_index & 1) {
            relevant_occupancy |= (1ULL << sqs[sqs_index]);
        }
        occupancy_index >>= 1;
        sqs_index++;
    }

    return relevant_occupancy;
}

void init_bishop_relevant_occupancies() {
    for (int sq=0; sq<64; sq++) {
        int n_relevant_occupancies = bishop_n_relevant_occupancies[sq];
        for (int occupancy_index=0; occupancy_index < (1<<n_relevant_occupancies); occupancy_index++) {
            bishop_relevant_occupancies[sq][occupancy_index] = bishop_relevant_occupancy(sq, occupancy_index);
        } 
    }
}

void bishop_visual_test(int sq) {
    for (int i=0; i<(1<<bishop_n_relevant_occupancies[sq]); i++) {
        print_bitboard(bishop_relevant_occupancies[sq][i], sq);
        usleep(100000);
    }
}

void find_bishop_magic_numbers() {
    for (int sq=0; sq<1; sq++) {
        U64 mask = bishop_relevant_bits_masks[sq];
        int max_occupancy_index = 1<<bishop_n_relevant_occupancies[sq];
        print_bitboard(mask, sq);
        for (U64 i=0; i<100000000ULL; i++) {
            U64 magic_number_candidate = random_U64_few_bits();
            if (bit_count((magic_number_candidate * mask) & 0xff00000000000000) < 6) continue;
            // printf("candidate: %llu\n", magic_number_candidate);

            int fail = 0;
            U64 attack_table[512] = {0};
            for (int occupancy_index=0; occupancy_index<max_occupancy_index; occupancy_index++) {
                U64 relevant_occupancy = bishop_relevant_occupancies[sq][occupancy_index];
                int magic_index = (relevant_occupancy * magic_number_candidate) >> 55;
                if (attack_table[magic_index] == 0) {
                    attack_table[magic_index] = bishop_attacks_slow(sq, relevant_occupancy);
                }
                else if (attack_table[magic_index] != bishop_attacks_slow(sq, relevant_occupancy)) {
                    fail = 1;
                    break;
                }
            }
            if (!fail) {
                printf("%llx\n", magic_number_candidate);
                break;
            }
        }
    }
}

/*/////////////////////////////////////////////////////////////////////////////
                                   main
/*/////////////////////////////////////////////////////////////////////////////

void init() {
    init_bishop_masks();
    init_bishop_relevant_occupancies();
}

int main() {
    init();

    find_bishop_magic_numbers();
    
    // U64 mask = bishop_relevant_bits_masks[0];
    // U64 magic_number = 0x40040822862081ULL;
    // for (int occupancy_index=0; occupancy_index<4; occupancy_index++) {
    //     U64 relevant_occupancy = bishop_relevant_occupancies[0][occupancy_index];
    //     int magic_index = (int)((relevant_occupancy * magic_number) >> 58);
    //     print_bitboard(relevant_occupancy, a8);
    //     printf("%llu\n", magic_index);
    // }

    return 0;
}
