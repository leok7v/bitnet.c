#include "chat_template.h"
#include "tools.h"
#include "model.h"
#include "session.h"
#include "generate.h"
#include "tokenizer.h"
#include "sampler.h"
#include "gguf.h"
#include "threadpool.h"

#ifdef BN_ENABLE_METAL
#include "gpu_metal.h"
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOKENS               8192
#define MAX_GEN_PER_TURN         1024
#define MAX_TOOL_CALLS_PER_TURN  4
#define MAX_PARAMS_PER_TOOL_CALL 8
#define MAX_TURNS                8
#define MAX_MSGS                256
#define MAX_ROUNDS_PER_TURN      2

typedef struct {
    char *buf;
    size_t len, cap;
} ByteBuf;

static void bb_reserve(ByteBuf *b, size_t need) {
    if (need <= b->cap) return;
    size_t c = b->cap ? b->cap : 256;
    while (c < need) c *= 2;
    char *nb = realloc(b->buf, c);
    if (!nb) { fprintf(stderr, "bb_reserve oom\n"); exit(1); }
    b->buf = nb; b->cap = c;
}

static void bb_append(ByteBuf *b, const char *s) {
    size_t n = strlen(s);
    bb_reserve(b, b->len + n + 1);
    memcpy(b->buf + b->len, s, n);
    b->len += n;
    b->buf[b->len] = '\0';
}

static void bb_append_n(ByteBuf *b, const char *s, size_t n) {
    bb_reserve(b, b->len + n + 1);
    memcpy(b->buf + b->len, s, n);
    b->len += n;
    b->buf[b->len] = '\0';
}

static void bb_reset(ByteBuf *b) {
    b->len = 0;
    if (b->buf) b->buf[0] = '\0';
}

static void bb_free(ByteBuf *b) { free(b->buf); b->buf = NULL; b->len = b->cap = 0; }

static void bb_append_json_escaped(ByteBuf *b, const char *s) {
    if (!s) { bb_append(b, "null"); return; }
    bb_append(b, "\"");
    for (const char *p = s; *p; p++) {
        switch (*p) {
            case '"':  bb_append(b, "\\\""); break;
            case '\\': bb_append(b, "\\\\"); break;
            case '\n': bb_append(b, "\\n");  break;
            case '\r': bb_append(b, "\\r");  break;
            case '\t': bb_append(b, "\\t");  break;
            default:
                if ((unsigned char)*p < 0x20) {
                    char esc[8];
                    snprintf(esc, sizeof(esc), "\\u%04x", (unsigned char)*p);
                    bb_append(b, esc);
                } else {
                    bb_append_n(b, p, 1);
                }
        }
    }
    bb_append(b, "\"");
}

typedef struct {
    char *function_name;
    char *param_names[MAX_PARAMS_PER_TOOL_CALL];
    char *param_values[MAX_PARAMS_PER_TOOL_CALL];
    int n_params;
    char *result;
} ParsedToolCall;

static void parsed_tc_free(ParsedToolCall *tc) {
    free(tc->function_name);
    for (int i = 0; i < tc->n_params; i++) {
        free(tc->param_names[i]);
        free(tc->param_values[i]);
    }
    free(tc->result);
    memset(tc, 0, sizeof(*tc));
}

static int parse_tool_call(const char *block, size_t block_len,
                           ParsedToolCall *out) {
    memset(out, 0, sizeof(*out));
    const char *p = block;
    const char *end = block + block_len;
    const char *fn_open = "<function=";
    const char *fn = memmem(p, end - p, fn_open, strlen(fn_open));
    if (!fn) return -1;
    fn += strlen(fn_open);
    const char *fn_close = memchr(fn, '>', end - fn);
    if (!fn_close) return -1;
    out->function_name = strndup(fn, fn_close - fn);
    if (!out->function_name) return -1;

    p = fn_close + 1;
    const char *param_open = "<parameter=";
    const char *param_close = "</parameter>";
    while (out->n_params < MAX_PARAMS_PER_TOOL_CALL) {
        const char *po = memmem(p, end - p, param_open, strlen(param_open));
        if (!po) break;
        po += strlen(param_open);
        const char *gt = memchr(po, '>', end - po);
        if (!gt) break;
        char *key = strndup(po, gt - po);
        const char *vstart = gt + 1;
        if (*vstart == '\n') vstart++;
        const char *pc = memmem(vstart, end - vstart, param_close, strlen(param_close));
        if (!pc) { free(key); break; }
        const char *vend = pc;
        if (vend > vstart && vend[-1] == '\n') vend--;
        char *val = strndup(vstart, vend - vstart);
        // Tolerate the small-model malformation <parameter=name="value"> (kwarg
        // jammed into the name slot, empty body) instead of the well-formed
        // <parameter=name>value</parameter>. A real parameter name never holds
        // '=', so any '=' in the key marks this case: split and take the inline
        // value, stripping one layer of surrounding quotes.
        char *eq = key ? strchr(key, '=') : NULL;
        if (eq) {
            *eq = '\0';
            char *iv = eq + 1;
            size_t ivn = strlen(iv);
            if (ivn >= 2 && (iv[0] == '"' || iv[0] == '\'') && iv[ivn - 1] == iv[0]) {
                iv[ivn - 1] = '\0';
                iv++;
            }
            free(val);
            val = strdup(iv);
        }
        out->param_names[out->n_params] = key;
        out->param_values[out->n_params] = val;
        out->n_params++;
        p = pc + strlen(param_close);
    }
    return 0;
}

typedef struct {
    ByteBuf strpool;
    char *owned_strings[MAX_MSGS * 4];
    int n_owned;
} StringPool;

static char *pool_dup(StringPool *sp, const char *s) {
    if (!s) return NULL;
    char *d = strdup(s);
    if (!d) { fprintf(stderr, "pool_dup oom\n"); exit(1); }
    sp->owned_strings[sp->n_owned++] = d;
    return d;
}

static void pool_free(StringPool *sp) {
    for (int i = 0; i < sp->n_owned; i++) free(sp->owned_strings[i]);
    sp->n_owned = 0;
    bb_free(&sp->strpool);
}

static int find_complete_tool_call(const char *stream, size_t stream_len,
                                   size_t search_from,
                                   size_t *out_start, size_t *out_end) {
    const char *p = stream + search_from;
    size_t remaining = stream_len - search_from;
    const char *open = memmem(p, remaining, "<tool_call>", 11);
    if (!open) return 0;
    size_t pos_open = (size_t)(open - stream);
    size_t after_open = pos_open + 11;
    const char *q = stream + after_open;
    size_t rem2 = stream_len - after_open;
    const char *close = memmem(q, rem2, "</tool_call>", 12);
    if (!close) return 0;
    *out_start = after_open;
    *out_end = (size_t)(close - stream);
    return 1;
}

static void extract_reasoning_content(const char *emission,
                                      char **out_reasoning,
                                      char **out_content) {
    *out_reasoning = NULL;
    *out_content = NULL;
    if (!emission) { *out_content = strdup(""); return; }

    const char *open = strstr(emission, "<think>");
    const char *close = strstr(emission, "</think>");

    if (close) {
        const char *start = emission;
        if (open && open < close) start = open + strlen("<think>");
        if (*start == '\n') start++;
        size_t r_len = (size_t)(close - start);
        while (r_len > 0 && (start[r_len-1] == '\n' || start[r_len-1] == ' '))
            r_len--;
        *out_reasoning = strndup(start, r_len);
        const char *after = close + strlen("</think>");
        while (*after == '\n') after++;
        *out_content = strdup(after);
    } else if (open) {
        const char *start = open + strlen("<think>");
        if (*start == '\n') start++;
        size_t n = strlen(start);
        while (n > 0 && (start[n-1] == '\n' || start[n-1] == ' ')) n--;
        *out_reasoning = strndup(start, n);
        *out_content = strdup("");
    } else {
        *out_content = strdup(emission);
        *out_reasoning = NULL;
    }
}

static char *content_before_first_tool_call(const char *content) {
    if (!content) return strdup("");
    const char *p = strstr(content, "<tool_call>");
    if (!p) return strdup(content);
    size_t n = (size_t)(p - content);
    while (n > 0 && (content[n-1] == '\n' || content[n-1] == ' ')) n--;
    return strndup(content, n);
}

typedef struct {
    BnTplMessage msgs[MAX_MSGS];
    int n_msgs;
    BnTplToolCall msg_tool_calls[MAX_MSGS][MAX_TOOL_CALLS_PER_TURN];
    BnTplParam    msg_tc_params[MAX_MSGS][MAX_TOOL_CALLS_PER_TURN][MAX_PARAMS_PER_TOOL_CALL];

    void *snap;
    size_t snap_bytes;
    int pos_pre_turn;

    BnModel *model;
    BnTokenizer *tok;
    BnSession *sess;
    BnSampler sampler;

    /* Chat template rendering context (read once from GGUF after model load). */
    const char *chat_tmpl;
    const char *bos_token;
    const char *eos_token;
} Agent;

static int render_msgs(Agent *a, int enable_thinking, int add_gen_prompt,
                       const BnTplTool *tools[], int n_tools,
                       int *out_tokens, int max_tokens) {
    BnTplTool tool_arr[8];
    for (int i = 0; i < n_tools && i < 8; i++) tool_arr[i] = *tools[i];
    return bn_chat_template_encode(a->tok, a->chat_tmpl,
                                   a->msgs, a->n_msgs,
                                   tool_arr, n_tools,
                                   enable_thinking, add_gen_prompt,
                                   a->bos_token, a->eos_token,
                                   out_tokens, max_tokens);
}

static int prefill_range(Agent *a, const int *tokens, int n, int pos0) {
    if (n <= 0) return 0;
    float *lp = bn_prefill(a->model, a->sess, tokens, n, pos0, 0);
    return lp ? 0 : -1;
}

static int restore_and_extend(Agent *a, const int *full_tokens, int n_full) {
    if (bn_session_set_recurrent_state(a->sess, a->model,
                                       a->snap, a->snap_bytes) != 0) {
        fprintf(stderr, "restore failed\n");
        return -1;
    }
    bn_session_kv_truncate(a->sess, a->pos_pre_turn);
    if (n_full > a->pos_pre_turn) {
        if (prefill_range(a, full_tokens + a->pos_pre_turn,
                          n_full - a->pos_pre_turn, a->pos_pre_turn) != 0)
            return -1;
    }
    return 0;
}

static int snapshot_current(Agent *a) {
    return bn_session_get_recurrent_state(a->sess, a->model,
                                          a->snap, a->snap_bytes);
}

static int is_stop_token(const BnTokenizer *tok, int tid) {
    return (tid == tok->im_end_id || tid == tok->endoftext_id ||
            tid == tok->eos_id);
}

static int sample_and_advance(Agent *a, float *logits, int *pos) {
    int tid = bn_sampler_sample(&a->sampler, logits);
    if (prefill_range(a, &tid, 1, *pos) != 0) return -1;
    (*pos)++;
    return tid;
}

static int run_turn(Agent *a, const char *user_msg,
                    const BnTplTool *tools[], int n_tools,
                    int turn_index, ByteBuf *transcript) {
    StringPool *sp = (StringPool *)a->msgs[0].content;
    (void)sp;

    int user_idx = a->n_msgs;
    a->msgs[user_idx].role = BN_TPL_ROLE_USER;
    a->msgs[user_idx].content = strdup(user_msg);
    a->msgs[user_idx].reasoning = NULL;
    a->msgs[user_idx].tool_calls = NULL;
    a->msgs[user_idx].n_tool_calls = 0;
    a->n_msgs++;

    int prompt_tokens[MAX_TOKENS];
    int n_prompt = render_msgs(a, 1, 1,
                               tools, n_tools, prompt_tokens, MAX_TOKENS);
    if (n_prompt < 0) { fprintf(stderr, "render prompt failed\n"); return -1; }

    if (restore_and_extend(a, prompt_tokens, n_prompt) != 0) return -1;
    int pos = n_prompt;

    ByteBuf stream = {0};
    size_t scanned_to = 0;
    ParsedToolCall observed_tcs[MAX_TOOL_CALLS_PER_TURN];
    int n_observed_tcs = 0;
    int turn_done = 0;
    int gen_count = 0;
    int round_count = 0;

    float *logits = bn_prefill(a->model, a->sess, NULL, 0, pos, 0);
    {
        float *lp = bn_prefill(a->model, a->sess,
                               prompt_tokens + (n_prompt - 1), 1,
                               n_prompt - 1, 0);
        logits = lp;
    }

    while (gen_count < MAX_GEN_PER_TURN && !turn_done) {
        int tid = bn_sampler_sample(&a->sampler, logits);
        gen_count++;

        if (is_stop_token(a->tok, tid)) {
            size_t tc_start, tc_end;
            while (find_complete_tool_call(stream.buf ? stream.buf : "",
                                           stream.len,
                                           scanned_to, &tc_start, &tc_end) &&
                   n_observed_tcs < MAX_TOOL_CALLS_PER_TURN) {
                if (parse_tool_call(stream.buf + tc_start, tc_end - tc_start,
                                    &observed_tcs[n_observed_tcs]) == 0) {
                    n_observed_tcs++;
                }
                scanned_to = tc_end + 12;
            }
            if (n_observed_tcs == 0) {
                turn_done = 1;
                break;
            }
            for (int t = 0; t < n_observed_tcs; t++) {
                ParsedToolCall *tc = &observed_tcs[t];
                ToolArg args[MAX_PARAMS_PER_TOOL_CALL];
                for (int p = 0; p < tc->n_params; p++) {
                    args[p].name = tc->param_names[p];
                    args[p].value = tc->param_values[p];
                }
                tc->result = tools_execute(tc->function_name, args, tc->n_params);
                if (!tc->result) tc->result = strdup("error: tool execution failed");
                printf("[tool] %s -> %s\n", tc->function_name, tc->result);
                fflush(stdout);
            }
            round_count++;

            char *reasoning = NULL, *content = NULL;
            extract_reasoning_content(stream.buf ? stream.buf : "",
                                      &reasoning, &content);
            char *content_clean = content_before_first_tool_call(content);
            free(content);
            content = content_clean;

            int asst_idx = a->n_msgs;
            for (int t = 0; t < n_observed_tcs; t++) {
                ParsedToolCall *tc = &observed_tcs[t];
                BnTplToolCall *qc = &a->msg_tool_calls[asst_idx][t];
                qc->function_name = strdup(tc->function_name);
                qc->n_params = tc->n_params;
                for (int p = 0; p < tc->n_params; p++) {
                    a->msg_tc_params[asst_idx][t][p].name = strdup(tc->param_names[p]);
                    a->msg_tc_params[asst_idx][t][p].value = strdup(tc->param_values[p]);
                }
                qc->params = a->msg_tc_params[asst_idx][t];
            }
            a->msgs[asst_idx].role = BN_TPL_ROLE_ASSISTANT;
            a->msgs[asst_idx].content = content;
            a->msgs[asst_idx].reasoning = reasoning;
            a->msgs[asst_idx].tool_calls = a->msg_tool_calls[asst_idx];
            a->msgs[asst_idx].n_tool_calls = n_observed_tcs;
            a->n_msgs++;

            for (int t = 0; t < n_observed_tcs; t++) {
                int tool_idx = a->n_msgs;
                a->msgs[tool_idx].role = BN_TPL_ROLE_TOOL;
                a->msgs[tool_idx].content = strdup(observed_tcs[t].result);
                a->msgs[tool_idx].reasoning = NULL;
                a->msgs[tool_idx].tool_calls = NULL;
                a->msgs[tool_idx].n_tool_calls = 0;
                a->n_msgs++;
            }

            int cont_tokens[MAX_TOKENS];
            int enable_thinking_now = (round_count < MAX_ROUNDS_PER_TURN) ? 1 : 0;
            int n_cont = render_msgs(a, enable_thinking_now, 1,
                                     tools, n_tools, cont_tokens, MAX_TOKENS);
            if (n_cont < 0) {
                fprintf(stderr, "render continuation failed\n"); return -1;
            }
            if (n_cont > pos) {
                if (prefill_range(a, cont_tokens + pos, n_cont - pos, pos) != 0)
                    return -1;
                float *lp = bn_prefill(a->model, a->sess,
                                       cont_tokens + (n_cont - 1), 1,
                                       n_cont - 1, 0);
                logits = lp;
                pos = n_cont;
            }
            bb_reset(&stream);
            scanned_to = 0;
            for (int t = 0; t < n_observed_tcs; t++)
                parsed_tc_free(&observed_tcs[t]);
            n_observed_tcs = 0;
            continue;
        }

        const char *piece = bn_tokenizer_decode(a->tok, tid);
        if (piece) bb_append(&stream, piece);

        float *lp = bn_prefill(a->model, a->sess, &tid, 1, pos, 0);
        if (!lp) return -1;
        logits = lp;
        pos++;
    }

    if (stream.buf && stream.len > 0) {
        char *reasoning = NULL, *content = NULL;
        extract_reasoning_content(stream.buf, &reasoning, &content);
        int has_content = content && *content;
        int has_reasoning = reasoning && *reasoning;
        if (has_content || has_reasoning) {
            int asst_idx = a->n_msgs;
            a->msgs[asst_idx].role = BN_TPL_ROLE_ASSISTANT;
            a->msgs[asst_idx].content = content;
            a->msgs[asst_idx].reasoning = reasoning;
            a->msgs[asst_idx].tool_calls = NULL;
            a->msgs[asst_idx].n_tool_calls = 0;
            a->n_msgs++;
        } else {
            free(reasoning);
            free(content);
        }
    }

    if (turn_index > 0) bb_append(transcript, ",\n");
    bb_append(transcript, "    {\n");
    bb_append(transcript, "      \"user\": ");
    bb_append_json_escaped(transcript, user_msg);
    bb_append(transcript, ",\n      \"tool_calls\": [");
    int wrote_tc = 0;
    for (int i = user_idx + 1; i < a->n_msgs; i++) {
        if (a->msgs[i].role == BN_TPL_ROLE_ASSISTANT &&
            a->msgs[i].n_tool_calls > 0) {
            for (int t = 0; t < a->msgs[i].n_tool_calls; t++) {
                const BnTplToolCall *tc = &a->msgs[i].tool_calls[t];
                bb_append(transcript, wrote_tc ? ", " : "");
                bb_append(transcript, "{\"name\": ");
                bb_append_json_escaped(transcript, tc->function_name);
                bb_append(transcript, ", \"args\": {");
                for (int p = 0; p < tc->n_params; p++) {
                    bb_append(transcript, p ? ", " : "");
                    bb_append_json_escaped(transcript, tc->params[p].name);
                    bb_append(transcript, ": ");
                    bb_append_json_escaped(transcript, tc->params[p].value);
                }
                bb_append(transcript, "}}");
                wrote_tc++;
            }
        }
    }
    bb_append(transcript, "],\n      \"tool_results\": [");
    int wrote_tr = 0;
    for (int i = user_idx + 1; i < a->n_msgs; i++) {
        if (a->msgs[i].role == BN_TPL_ROLE_TOOL) {
            bb_append(transcript, wrote_tr ? ", " : "");
            bb_append_json_escaped(transcript, a->msgs[i].content);
            wrote_tr++;
        }
    }
    const char *final_response = "";
    for (int i = a->n_msgs - 1; i >= user_idx + 1; i--) {
        if (a->msgs[i].role == BN_TPL_ROLE_ASSISTANT) {
            final_response = a->msgs[i].content ? a->msgs[i].content : "";
            break;
        }
    }
    bb_append(transcript, "],\n      \"final_response\": ");
    bb_append_json_escaped(transcript, final_response);
    bb_append(transcript, "\n    }");
    fflush(stdout);

    bb_free(&stream);

    /* Re-render the full conversation (no clean_history concept in bn_chat_template_encode)
       and snapshot so the next turn starts from a consistent KV position. */
    int clean_tokens[MAX_TOKENS];
    int n_clean = render_msgs(a, 1, 0,
                              tools, n_tools, clean_tokens, MAX_TOKENS);
    if (n_clean < 0) {
        fprintf(stderr, "render clean history failed\n"); return -1;
    }
    if (restore_and_extend(a, clean_tokens, n_clean) != 0) return -1;
    if (snapshot_current(a) != 0) {
        fprintf(stderr, "snapshot failed\n"); return -1;
    }
    a->pos_pre_turn = n_clean;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <model.gguf> [--out transcript.json] [--metal]\n",
                argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *transcript_path = "tr.json";
    int use_metal = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) transcript_path = argv[++i];
        else if (strcmp(argv[i], "--metal") == 0) use_metal = 1;
    }

    BnGGUFFile *gf = bn_gguf_open_file(model_path);
    if (!gf) { fprintf(stderr, "gguf open failed\n"); return 1; }
    BnModel model;
    if (bn_model_load(&model, gf, MAX_TOKENS, 0, 0) != 0) {
        fprintf(stderr, "model load failed\n"); return 1;
    }
    BnTokenizer tok;
    if (bn_tokenizer_init(&tok, gf) != 0) {
        fprintf(stderr, "tokenizer init failed\n"); return 1;
    }
    if (model.config.full_attn_interval <= 0) {
        fprintf(stderr, "model is not hybrid GDN; agent requires Qwen3.5\n");
        return 1;
    }

    /* Read Jinja chat template and BOS/EOS tokens from the GGUF once. */
    const char *chat_tmpl = bn_gguf_get_str(gf, "tokenizer.chat_template");
    if (!chat_tmpl) {
        fprintf(stderr, "model has no tokenizer.chat_template\n");
        return 1;
    }
    const char *bos_token = (tok.bos_id >= 0 && tok.bos_id < tok.vocab_size)
                            ? tok.vocab[tok.bos_id] : NULL;
    const char *eos_token = (tok.eos_id >= 0 && tok.eos_id < tok.vocab_size)
                            ? tok.vocab[tok.eos_id] : NULL;

    /* Spin up a CPU thread pool: use half the performance cores minus one spare
       so the main thread and OS stay responsive while the model runs. */
    int n_workers = 0;
#if defined(__APPLE__)
    {
        int n = 0;
        size_t l = sizeof(n);
        if (sysctlbyname("hw.perflevel0.logicalcpu", &n, &l, NULL, 0) == 0 && n > 1)
            n_workers = n - 1;
    }
#else
    {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        if (n > 1) n_workers = (int)(n / 2);
    }
#endif
    bn_model_set_thread_pool(&model, bn_tp_create(n_workers), 1);

#ifdef BN_ENABLE_METAL
    if (use_metal) {
        BnGPUBackend *gpu = bn_gpu_metal_create("shaders/metal/");
        if (!gpu) {
            fprintf(stderr, "Metal init failed; using CPU\n");
        } else if (bn_model_upload_weights(&model, gpu) != 0) {
            fprintf(stderr, "Metal weight upload failed; using CPU\n");
            bn_gpu_metal_destroy(gpu);
        } else if (gpu->init_activations &&
                   gpu->init_activations(gpu->ctx, &model.config) != 0) {
            fprintf(stderr, "Metal init_activations failed; using CPU\n");
            bn_gpu_metal_destroy(gpu);
        } else {
            printf("[agent] Metal backend attached\n");
        }
    }
#else
    if (use_metal) {
        fprintf(stderr, "--metal requires BN_ENABLE_METAL=1 build; using CPU\n");
    }
#endif

    Agent a = {0};
    a.model = &model;
    a.tok = &tok;
    a.sess = bn_session_create(&model, NULL);
    if (!a.sess) { fprintf(stderr, "session failed\n"); return 1; }
    a.snap_bytes = bn_session_recurrent_state_bytes(&model);
    a.snap = malloc(a.snap_bytes);
    if (!a.snap) { fprintf(stderr, "snap alloc failed\n"); return 1; }
    a.chat_tmpl = chat_tmpl;
    a.bos_token = bos_token;
    a.eos_token = eos_token;

    // Unsloth's recommended Qwen3.5 sampling (non-thinking text tasks):
    //   temperature=1.0, top_p=1.0, top_k=20, min_p=0.0,
    //   presence_penalty=2.0, repetition_penalty=1.0
    //   (https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)
    // This sampler wires temp/top_p/repeat_penalty only; top_k, min_p, and
    // presence_penalty are not implemented, so we set the three we have and
    // leave the rest. Override any with BN_AGENT_TEMP / BN_AGENT_TOPP /
    // BN_AGENT_SEED (e.g. BN_AGENT_TEMP=0 for a deterministic greedy run).
    float agent_temp = 1.0f;
    float agent_topp = 1.0f;
    uint64_t agent_seed = 0;
    const char *env_temp = getenv("BN_AGENT_TEMP");
    const char *env_topp = getenv("BN_AGENT_TOPP");
    const char *env_seed = getenv("BN_AGENT_SEED");
    if (env_temp) agent_temp = (float)atof(env_temp);
    if (env_topp) agent_topp = (float)atof(env_topp);
    if (env_seed) agent_seed = (uint64_t)atoll(env_seed);
    bn_sampler_init(&a.sampler, model.config.vocab_size,
                    agent_temp, agent_topp, agent_seed);

    const BnTplTool *tools[8] = {
        tools_get_datetime_def(), tools_read_file_def(),
        tools_file_glob_search_def(), tools_grep_search_def(),
        tools_exec_shell_command_def(), tools_write_file_def(),
        tools_edit_file_def(), tools_apply_diff_def()
    };
    int n_tools = 8;
    a.msgs[0].role = BN_TPL_ROLE_SYSTEM;
    a.msgs[0].content = strdup(
        "You are a coding agent working in the current project directory. "
        "You have file and shell tools: get_datetime, read_file, "
        "file_glob_search, grep_search, exec_shell_command, write_file, "
        "edit_file, apply_diff. Call a tool to inspect or modify files, then "
        "reply with a short final answer after the tool round-trip.");
    a.n_msgs = 1;

    a.pos_pre_turn = 0;
    if (snapshot_current(&a) != 0) {
        fprintf(stderr, "initial snapshot failed\n"); return 1;
    }

    ByteBuf transcript = {0};
    const char *model_name = strrchr(model_path, '/');
    model_name = model_name ? model_name + 1 : model_path;
    char model_name_buf[128];
    {
        size_t mn = strlen(model_name);
        if (mn >= sizeof(model_name_buf)) mn = sizeof(model_name_buf) - 1;
        memcpy(model_name_buf, model_name, mn);
        model_name_buf[mn] = '\0';
        char *dot = strrchr(model_name_buf, '.');
        if (dot) *dot = '\0';
    }
    bb_append(&transcript, "{\n  \"model\": \"");
    bb_append(&transcript, model_name_buf);
    bb_append(&transcript, "\",\n  \"turns\": [\n");

    const char *turn_prompts[] = {
        "What is the current date and time?",
        "List every file in the current working folder (path '.').",
        "Show the contents of the file notes.txt.",
        "Which lines of notes.txt contain the word TODO?",
        "Run the shell command 'ls -la' and show its output.",
        "Write a file named report.txt containing a one-sentence summary "
        "of notes.txt.",
        "Append the line 'reviewed: yes' to report.txt.",
        "Apply a unified diff to report.txt that changes 'reviewed: yes' "
        "to 'reviewed: done'.",
    };
    int n_turns = (int)(sizeof(turn_prompts) / sizeof(turn_prompts[0]));

    for (int t = 0; t < n_turns; t++) {
        printf("\n========== turn %d: %s\n", t + 1, turn_prompts[t]);
        if (run_turn(&a, turn_prompts[t], tools, n_tools, t, &transcript) != 0) {
            fprintf(stderr, "turn %d failed\n", t + 1);
            break;
        }
    }

    bb_append(&transcript, "\n  ]\n}\n");

    FILE *fp = fopen(transcript_path, "w");
    if (fp) {
        fwrite(transcript.buf, 1, transcript.len, fp);
        fclose(fp);
        printf("\ntranscript: %s\n", transcript_path);
    } else {
        fprintf(stderr, "could not open %s for write\n", transcript_path);
    }

    bb_free(&transcript);
    free(a.snap);
    bn_session_free(a.sess, NULL);
    bn_tokenizer_free(&tok);
    bn_model_free(&model);
    bn_gguf_free(gf);
    return 0;
}
