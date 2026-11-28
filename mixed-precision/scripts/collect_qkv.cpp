// Calibration-only executable. Private model tensors identify the first attention input.
#include "whisper.cpp"
// Declared in examples/common-whisper.h, which also declares to_timestamp.
bool read_audio_data(const std::string &, std::vector<float> &,
                     std::vector<std::vector<float>> &, bool);
#include <iomanip>
#include <iostream>

struct moments {
    ggml_tensor * norm_bias;
    int frames;
    int observed = 0;
    std::vector<double> sum_squares;
};

static bool observe(ggml_tensor * node, bool ask, void * user_data) {
    auto & stats = *static_cast<moments *>(user_data);
    if (ask) {
        return node->op == GGML_OP_ADD && node->src[1] == stats.norm_bias;
    }
    GGML_ASSERT(node->type == GGML_TYPE_F32 && ggml_is_contiguous(node));
    GGML_ASSERT(stats.frames <= node->ne[1] && (size_t)node->ne[0] == stats.sum_squares.size());
    std::vector<float> x(stats.frames * node->ne[0]);
    ggml_backend_tensor_get(node, x.data(), 0, x.size() * sizeof(float));
    for (size_t i = 0; i < x.size(); ++i) {
        stats.sum_squares[i % node->ne[0]] += double(x[i]) * x[i];
    }
    stats.observed++;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "usage: collect-qkv fp16.bin clip.wav [clip.wav ...]\n";
        return 1;
    }
    ggml_backend_load_all();
    auto params = whisper_context_default_params();
    params.use_gpu = false;
    auto * ctx = whisper_init_from_file_with_params(argv[1], params);
    if (!ctx) {
        return 1;
    }
    moments stats{ctx->model.layers_encoder[0].attn_ln_0_b, 0, 0,
                  std::vector<double>(ctx->model.hparams.n_audio_state, 0.0)};
    ggml_backend_sched_set_eval_callback(ctx->state->sched_encode.sched, observe, &stats);
    int total_frames = 0;
    for (int i = 2; i < argc; ++i) {
        std::vector<float> pcm;
        std::vector<std::vector<float>> channels;
        if (!read_audio_data(argv[i], pcm, channels, false) || pcm.empty() || pcm.size() > 30*16000) {
            std::cerr << "Expected a non-empty calibration clip of at most 30 s: " << argv[i] << "\n";
            return 1;
        }
        stats.frames = (pcm.size() + 319) / 320; // Encoder positions are 20 ms apart.
        // Extra frames retain the native convolutional receptive field at the right edge.
        ctx->state->exp_n_audio_ctx = std::min(1500, stats.frames + 16);
        if (whisper_pcm_to_mel(ctx, pcm.data(), pcm.size(), 4) ||
            whisper_encode(ctx, 0, 4) || stats.observed != i - 1) {
            std::cerr << "Calibration activation missing for " << argv[i] << "\n";
            return 1;
        }
        total_frames += stats.frames;
    }
    std::cout << total_frames << "\n" << std::setprecision(17);
    for (double sum : stats.sum_squares) {
        std::cout << sum / total_frames << "\n";
    }
    whisper_free(ctx);
}
