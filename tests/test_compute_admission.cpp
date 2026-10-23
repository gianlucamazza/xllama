#include <doctest/doctest.h>

#include "xllama/compute_admission.h"

using xllama::ComputeJobAdmission;

TEST_CASE("compute admission grants one permit and releases it on destruction") {
    ComputeJobAdmission admission;
    auto first = admission.try_acquire();
    REQUIRE(first);
    CHECK_FALSE(admission.try_acquire());

    first.reset();
    CHECK(admission.try_acquire());
}

TEST_CASE("compute admission stays held while an async owner retains the permit") {
    ComputeJobAdmission admission;
    auto worker = admission.try_acquire();
    REQUIRE(worker);
    auto completion = worker;
    worker.reset();
    CHECK_FALSE(admission.try_acquire());

    completion.reset();
    CHECK(admission.try_acquire());
}
