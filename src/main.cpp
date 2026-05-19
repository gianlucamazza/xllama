#include "xllama/cli.h"
#include "xllama/inference.h"

int main(int argc, char** argv) {
    xllama::InferenceParams params;
    if (!xllama::parse_cli_args(argc, argv, params))
        return 1;

    auto res = xllama::run_inference(params);
    return res.success ? 0 : 1;
}
