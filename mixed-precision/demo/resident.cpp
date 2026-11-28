// Small C interface to the pinned libwhisper. Each PCM window starts with an empty text context.
#include "whisper.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct resident {
    whisper_context * ctx;
    std::string text;
};

extern "C" resident * resident_open(const char * model) {
    auto params = whisper_context_default_params();
    params.use_gpu = false;
    params.flash_attn = true;
    auto * ctx = whisper_init_from_file_with_params_no_state(model, params);
    return ctx ? new resident{ctx, {}} : nullptr;
}

extern "C" const char * resident_decode(resident * model, const int16_t * samples, int n,
                                       const char * language, int threads, int context) {
    std::vector<float> pcm(n);
    for (int i = 0; i < n; ++i) pcm[i] = samples[i] / 32768.0f;
    auto params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language = language;
    params.n_threads = threads;
    params.audio_ctx = context;
    params.no_context = true;
    params.no_timestamps = true;
    params.greedy.best_of = 1;
    params.beam_search.beam_size = 1;
    params.temperature = 0;
    params.temperature_inc = 0;
    params.print_progress = false;
    params.print_timestamps = false;
    std::unique_ptr<whisper_state, decltype(&whisper_free_state)> state(
        whisper_init_state(model->ctx), whisper_free_state);
    if (!state || whisper_full_with_state(model->ctx, state.get(), params, pcm.data(), n) != 0) return nullptr;
    model->text.clear();
    for (int i = 0; i < whisper_full_n_segments_from_state(state.get()); ++i) {
        model->text += whisper_full_get_segment_text_from_state(state.get(), i);
        model->text += ' ';
    }
    return model->text.c_str();
}

extern "C" void resident_close(resident * model) {
    whisper_free(model->ctx);
    delete model;
}
