// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/prompt_budget.h"
#include <climits>
#include <iostream>
int main() {
    xllama::ChatFormat fmt;
    auto fit = xllama::fit_prompt(fmt, "", {}, "hello", 2048, INT_MAX,
                                  [](const std::string&) { return 10; });
    std::cout << "fits=" << fit.fits << " tokens=" << fit.n_tokens << "\n";
    return fit.fits ? 1 : 0;
}
