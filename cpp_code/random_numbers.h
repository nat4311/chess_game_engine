#pragma once

#include "engine.h"
#include <stdlib.h>

U64 random_U64() {
    U64 a, b, c , d;
    a = (U64)(random() & 0xffff);
    b = (U64)(random() & 0xffff);
    c = (U64)(random() & 0xffff);
    d = (U64)(random() & 0xffff);
    return a | (b<<16) | (c<<32) | (d<<48);
}
U64 random_U64_few_bits() {
    return random_U64() & random_U64() & random_U64();
}

