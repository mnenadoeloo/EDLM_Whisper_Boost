// Use the pinned loader so the profiled weights have the same CPU buffer/repacking as Whisper.
#include "whisper.cpp"
#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: m3-matmul model.bin block-index\n";
        return 1;
    }
    ggml_backend_load_all();
    auto params = whisper_context_default_params();
    params.use_gpu = false;
    auto * model = whisper_init_from_file_with_params_no_state(argv[1], params);
    if (!model) return 1;
    auto cpu = ggml_backend_cpu_init();
    auto tp_params = ggml_threadpool_params_default(4);
    auto pool = ggml_threadpool_new(&tp_params);
    ggml_backend_cpu_set_threadpool(cpu, pool);
    ggml_backend_cpu_set_n_threads(cpu, 4);
    const std::string block = std::string("blocks.") + argv[2] + ".";
    struct workload { std::string role, name; std::vector<int> columns; };
    const std::vector<workload> cases = {
        {"enc_attn", "encoder." + block + "attn.query.weight", {768, 1024}},
        {"cross_kv", "decoder." + block + "cross_attn.key.weight", {768, 1024}},
        {"cross_token", "decoder." + block + "cross_attn.query.weight", {1, 4}},
        {"enc_up", "encoder." + block + "mlp.0.weight", {768}},
        {"enc_down", "encoder." + block + "mlp.2.weight", {768}},
        {"dec_up", "decoder." + block + "mlp.0.weight", {1, 4}},
        {"dec_down", "decoder." + block + "mlp.2.weight", {1, 4}},
        {"head", "decoder.token_embedding.weight", {1, 4}},
    };
    for (const auto & item : cases) {
        auto * w = model->model.tensors.at(item.name);
        for (int n : item.columns) {
            auto * ctx = ggml_init({2 << 20, nullptr, true});
            auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w->ne[0], n);
            auto * y = ggml_mul_mat(ctx, w, x);
            auto * graph = ggml_new_graph_custom(ctx, 8, false);
            ggml_build_forward_expand(graph, y);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
            std::vector<float> input(ggml_nelements(x));
            for (size_t i = 0; i < input.size(); ++i) input[i] = (int(i % 127) - 63) / 64.0f;
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            for (int rep = -2; rep < 7; ++rep) {
                const auto start = std::chrono::steady_clock::now();
                if (ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS) return 2;
                const double us = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start).count();
                float check = 0.0f;
                ggml_backend_tensor_get(y, &check, 0, sizeof(check));
                if (!std::isfinite(check) || check == 0.0f) return 3;
                if (rep >= 0) {
                    std::cout << item.role << ',' << w->ne[0] << ',' << w->ne[1] << ',' << n << ','
                              << ggml_backend_buffer_name(w->buffer) << ',' << rep << ','
                              << std::fixed << std::setprecision(3) << us << '\n';
                }
            }
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(cpu);
    ggml_threadpool_free(pool);
    whisper_free(model);
}
