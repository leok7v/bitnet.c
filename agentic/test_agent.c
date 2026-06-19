#include "chat_template.h"
#include "tools.h"
#include "model.h"
#include "session.h"
#include "generate.h"
#include "tokenizer.h"
#include "sampler.h"
#include "grammar.h"
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
#include <unistd.h>     /* chdir, getcwd */
#include <sys/stat.h>   /* mkdir */
#include <errno.h>
#include <time.h>       /* clock_gettime */
#ifdef __APPLE__
#include <mach/mach.h>  /* task_info -> current resident size */
#else
#include <sys/resource.h>
#endif

#define MAX_TOKENS               65536
#define MAX_GEN_PER_TURN         1024
#define MAX_TOOL_CALLS_PER_TURN  4
#define MAX_PARAMS_PER_TOOL_CALL 8
#define MAX_TURNS                8
#define MAX_MSGS                256
#define MAX_ROUNDS_PER_TURN      2
#define MAX_IDENTICAL_ROUNDS     3   /* loop-breaker: stop after N repeats */
#define MAX_TOTAL_ROUNDS         8   /* loop-breaker: hard cap on tool rounds */

/* ---- lightweight per-turn instrumentation ----------------------------------
 * pp = prompt processing (bulk prefill of the user turn + tool-result
 * continuations); tg = token generation (per-token decode). We time the two
 * primitives separately and accumulate per turn, then print a readable table.
 * RSS is the current resident footprint sampled at the end of each turn, so the
 * table shows how RSS tracks context growth. The end-of-turn "clean re-render"
 * snapshot prefill is deliberately NOT counted (accumulators reset per turn). */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static size_t cur_rss_bytes(void) {
#ifdef __APPLE__
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &cnt) == KERN_SUCCESS)
        return (size_t)info.resident_size;
    return 0;
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        return (size_t)ru.ru_maxrss * 1024; /* Linux reports KiB */
    return 0;
#endif
}

/* phys_footprint is the memory the process is actually charged for (anonymous +
 * wired GPU buffers), excluding clean mmap'd file pages -- a truer figure than
 * resident_size for an mmap-weights inference engine. */

static size_t cur_footprint_bytes(void) {
    size_t bytes = 0;
#ifdef __APPLE__
    task_vm_info_data_t vm;
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  (task_info_t)&vm, &cnt) == KERN_SUCCESS)
        bytes = (size_t)vm.phys_footprint;
#endif
    return bytes;
}

static double cur_cpu_ms(void) {
    struct rusage ru;
    double ms = 0.0;
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        ms = (double)(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1000.0 +
             (double)(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1000.0;
    return ms;
}

/* macOS system memory-pressure level: 1 normal, 2 warn, 4 critical (-1 if the
 * sysctl is unavailable). Process-local GPU utilization is not exposed by the
 * OS, so the table derives a GPU/wait share from off-CPU wall time instead. */

static int mem_pressure_level(void) {
    int level = -1;
#ifdef __APPLE__
    size_t sz = sizeof(level);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level",
                     &level, &sz, NULL, 0) != 0)
        level = -1;
#endif
    return level;
}

/* Integral GPU-active time (ms) from the Metal backend's command-buffer
 * GPUStartTime/GPUEndTime sum; per-turn delta gives a real GPU-busy share. */

static double cur_gpu_active_ms(const BnModel *model) {
    double ms = 0.0;
#ifdef BN_ENABLE_METAL
    ms = bn_gpu_metal_active_ms(bn_model_gpu(model));
#else
    (void)model;
#endif
    return ms;
}

typedef struct {
    int    ctx_in;     /* context tokens carried into the turn */
    int    ctx_out;    /* context tokens after generation */
    long   pp_tok;     /* tokens bulk-prefilled this turn */
    double pp_ms;
    long   tg_tok;     /* tokens generated (decoded) this turn */
    double tg_ms;
    int    rounds;     /* tool round-trips */
    int    tool_ok;
    int    tool_err;
    size_t rss;        /* resident bytes at end of turn */
    size_t footprint;  /* phys_footprint bytes at end of turn */
    double wall_ms;    /* wall time spent in the turn */
    double cpu_ms;     /* process CPU time (user+sys) spent in the turn */
    double gpu_ms;     /* GPU-active time (command-buffer windows) in the turn */
    int    pressure;   /* system memory-pressure level at end of turn */
} TurnMetrics;

static double g_pp_ms, g_tg_ms;   /* per-turn accumulators (reset each turn) */
static long   g_pp_tok, g_tg_tok;
static TurnMetrics g_metrics[MAX_TURNS];
static int    g_n_metrics;

/* --vv / --verbose: echo the full rendered prompt, every generated token, and
 * tool args/results to stdout so the whole model interaction is visible. */
static int    g_verbose;

static void print_metrics_table(const char *model_name) {
    printf("\n========== metrics: %s ==========\n", model_name);
    printf("%-4s %7s %8s %7s %9s %7s %9s %5s %7s %8s %7s %5s %6s %5s\n",
           "turn", "ctx_in", "ctx_out", "pp_tok", "pp_tok/s",
           "tg_tok", "tg_tok/s", "rnds", "tools", "rss_MB",
           "fp_MB", "cpu%", "gpu%", "press");
    double tot_pp_ms = 0, tot_tg_ms = 0;
    long tot_pp = 0, tot_tg = 0;
    size_t peak = 0, peak_fp = 0;
    for (int i = 0; i < g_n_metrics; i++) {
        TurnMetrics *m = &g_metrics[i];
        double pps = m->pp_ms > 0 ? (double)m->pp_tok * 1000.0 / m->pp_ms : 0;
        double tps = m->tg_ms > 0 ? (double)m->tg_tok * 1000.0 / m->tg_ms : 0;
        double cpu_pct = m->wall_ms > 0 ? m->cpu_ms * 100.0 / m->wall_ms : 0;
        double gpu_pct = m->wall_ms > 0 ? m->gpu_ms * 100.0 / m->wall_ms : 0;
        char tcs[16];
        snprintf(tcs, sizeof tcs, "%d/%d", m->tool_ok, m->tool_err);
        printf("%-4d %7d %8d %7ld %9.1f %7ld %9.1f %5d %7s %8.0f "
               "%7.0f %5.0f %6.0f %5d\n",
               i + 1, m->ctx_in, m->ctx_out, m->pp_tok, pps,
               m->tg_tok, tps, m->rounds, tcs, (double)m->rss / 1048576.0,
               (double)m->footprint / 1048576.0, cpu_pct, gpu_pct, m->pressure);
        /* compact CSV for cross-model aggregation (grep '^#M,') */
        printf("#M,%s,%d,%d,%d,%ld,%.1f,%ld,%.1f,%d,%d,%d,%.0f,%.0f,%.0f,%.0f,%d\n",
               model_name, i + 1, m->ctx_in, m->ctx_out, m->pp_tok, pps,
               m->tg_tok, tps, m->rounds, m->tool_ok, m->tool_err,
               (double)m->rss / 1048576.0, (double)m->footprint / 1048576.0,
               cpu_pct, gpu_pct, m->pressure);
        tot_pp_ms += m->pp_ms; tot_tg_ms += m->tg_ms;
        tot_pp += m->pp_tok; tot_tg += m->tg_tok;
        if (m->rss > peak) peak = m->rss;
        if (m->footprint > peak_fp) peak_fp = m->footprint;
    }
    double opps = tot_pp_ms > 0 ? (double)tot_pp * 1000.0 / tot_pp_ms : 0;
    double otps = tot_tg_ms > 0 ? (double)tot_tg * 1000.0 / tot_tg_ms : 0;
    int final_ctx = g_n_metrics > 0 ? g_metrics[g_n_metrics - 1].ctx_out : 0;
    printf("%-4s %7s %8d %7ld %9.1f %7ld %9.1f %5s %7s %8.0f %7.0f %5s %6s %5s\n",
           "ALL", "-", final_ctx, tot_pp, opps, tot_tg, otps, "-", "-",
           (double)peak / 1048576.0, (double)peak_fp / 1048576.0, "-", "-", "-");
    printf("(pp=prefill, tg=token-gen; rss/fp=resident/phys-footprint MB per "
           "turn end; cpu%%=process CPU/wall (sums all cores), gpu%%=GPU-active "
           "wall fraction (Metal cmd-buffer GPU time); press=mem-pressure 1/2/4; "
           "peak rss=%.0f fp=%.0f MB, final ctx=%d tok)\n",
           (double)peak / 1048576.0, (double)peak_fp / 1048576.0, final_ctx);
}

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

/* Strip one layer of matching surrounding quotes, in place. */
static void unquote(char *s) {
    size_t n = strlen(s);
    if (n >= 2 && (s[0] == '"' || s[0] == '\'') && s[n - 1] == s[0]) {
        memmove(s, s + 1, n - 2);
        s[n - 2] = '\0';
    }
}

/* Parse one <tool_call> body. Qwen3.5 sizes emit several malformed variants;
 * this tolerates all observed shapes:
 *   - function name:  <function=NAME>  and the <fuction=NAME> typo (2B)
 *   - parameters:     <parameter=KEY>VALUE</parameter>   (Hermes / spec)
 *                     <KEY>VALUE</parameter>             (bare open, 4B)
 *                     <KEY>VALUE</KEY>                   (fully bare, 0.8B)
 *                     <KEY=VALUE> or <KEY="VALUE">       (attribute form, 2B)
 * Strategy: locate the function opener, then walk opening tags. A non-closing,
 * non-structural tag introduces a parameter; if it contains '=' the value is
 * inline (attribute form), otherwise the value is the body up to the matching
 * close (</KEY> or </parameter>), or the next '<' if neither close is present.
 * Preferring the explicit close lets values legitimately contain '<'. */
static int parse_tool_call(const char *block, size_t block_len,
                           ParsedToolCall *out) {
    memset(out, 0, sizeof(*out));
    const char *p = block;
    const char *end = block + block_len;

    /* Function opener: accept the common "fuction" misspelling too. */
    const char *fn = memmem(p, end - p, "<function=", 10);
    size_t fn_open_len = 10;
    if (!fn) { fn = memmem(p, end - p, "<fuction=", 9); fn_open_len = 9; }
    if (!fn) return -1;
    fn += fn_open_len;
    const char *fn_close = memchr(fn, '>', end - fn);
    if (!fn_close) return -1;
    size_t name_len = (size_t)(fn_close - fn);
    while (name_len > 0 && (fn[name_len - 1] == '/' || fn[name_len - 1] == ' '))
        name_len--; /* tolerate <function=name/> and trailing space */
    out->function_name = strndup(fn, name_len);
    if (!out->function_name) return -1;

    p = fn_close + 1;
    while (out->n_params < MAX_PARAMS_PER_TOOL_CALL) {
        const char *lt = memchr(p, '<', end - p);
        if (!lt) break;
        const char *gt = memchr(lt, '>', end - lt);
        if (!gt) break;
        size_t taglen = (size_t)(gt - (lt + 1));
        const char *tag = lt + 1;
        /* Skip closing tags and the function/tool_call structural tags. */
        if (taglen == 0 || tag[0] == '/' ||
            (taglen >= 8 && strncmp(tag, "function", 8) == 0) ||
            (taglen >= 7 && strncmp(tag, "fuction", 7) == 0) ||
            (taglen >= 9 && strncmp(tag, "tool_call", 9) == 0)) {
            p = gt + 1;
            continue;
        }
        char *key = strndup(tag, taglen);
        if (!key) break;
        if (strncmp(key, "parameter=", 10) == 0)
            memmove(key, key + 10, strlen(key + 10) + 1);

        char *eq = strchr(key, '=');
        if (eq) {
            /* Attribute form <KEY=VALUE>: value is inline, no body/close. */
            *eq = '\0';
            char *val = strdup(eq + 1);
            if (val) unquote(val);
            p = gt + 1;
            if (key[0] && val) {
                out->param_names[out->n_params] = key;
                out->param_values[out->n_params] = val;
                out->n_params++;
            } else { free(key); free(val); }
            continue;
        }

        /* Bare/Hermes form: value is the body up to the matching close. Prefer
         * </KEY> or </parameter> (so values may contain '<'); else next '<'. */
        const char *vstart = gt + 1;
        if (vstart < end && *vstart == '\n') vstart++;
        char close_tag[80];
        int cn = snprintf(close_tag, sizeof close_tag, "</%s>", key);
        const char *pc = NULL;
        if (cn > 0 && (size_t)cn < sizeof close_tag)
            pc = memmem(vstart, end - vstart, close_tag, (size_t)cn);
        if (!pc) pc = memmem(vstart, end - vstart, "</parameter>", 12);
        const char *vend = pc ? pc : memchr(vstart, '<', end - vstart);
        if (!vend) vend = end;
        const char *ve = vend;
        while (ve > vstart && (ve[-1] == '\n' || ve[-1] == ' ')) ve--;
        char *val = strndup(vstart, (size_t)(ve - vstart));
        if (key[0] && val) {
            out->param_names[out->n_params] = key;
            out->param_values[out->n_params] = val;
            out->n_params++;
        } else { free(key); free(val); }
        /* Advance past the body and an immediately following close tag. */
        p = vend;
        if (p < end && p[0] == '<') {
            const char *cgt = memchr(p, '>', end - p);
            if (cgt && p + 1 < end && p[1] == '/') p = cgt + 1;
        }
    }
    return 0;
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

/* Constrained decoding (--grammar): mask logits so the model can only emit
 * well-formed tool-call markup. The grammar forces the literal structural tags
 * (<tool_call>, <function=, <parameter=, the close tags) while leaving the
 * function name and each parameter name/value FREE. It is armed only inside a
 * <tool_call> block, so reasoning and final answers are unconstrained. The mask
 * no-ops when the grammar is dead so it never forbids the whole vocabulary. */
typedef struct {
    BnGrammar *g;
    const BnGrammarVocab *v;
} GrammarMaskCtx;

static void grammar_mask_cb(void *ctx, float *logits, int vocab_size) {
    GrammarMaskCtx *c = (GrammarMaskCtx *)ctx;
    (void)vocab_size;
    if (!bn_grammar_dead(c->g)) bn_grammar_mask_logits(c->g, c->v, logits);
}

/* Tool-call skeleton: <tool_call> WS <function=NAME> ( WS <parameter=PNAME>
 * VALUE </parameter> )* WS </function> WS </tool_call>. WS is (any byte except
 * '<')*, which absorbs the inter-tag newlines and also matches the zero-space
 * inline form. Parameters are zero-or-more so a no-arg call (get_datetime) emits
 * </function> straight after <function=NAME>. Names run until '>', values until
 * '<' (so a value may contain '>' but not '<' -- fine for paths/queries; a
 * literal '<' inside a value, e.g. embedded markup, is the known limitation). */
static int build_tool_call_grammar(BnGBuilder *b) {
    bn_gb_reset(b);
    bn_gb_lit(b, "<tool_call>");
    bn_gb_until_char(b, '<');                   /* WS before <function= */
    bn_gb_lit(b, "<function=");
    bn_gb_until_char(b, '>');                   /* function name, free */
    bn_gb_lit(b, ">");
    int loop = b->n;
    bn_gb_until_char(b, '<');                   /* WS before the next tag */
    int sp = bn_gb_emit(b, BN_G_SPLIT, 0, 0, -1, 0);  /* param | end, alt patched */
    bn_gb_lit(b, "<parameter=");
    bn_gb_until_char(b, '>');                   /* param name, free */
    bn_gb_lit(b, ">");
    bn_gb_until_char(b, '<');                   /* value, free until close */
    bn_gb_lit(b, "</parameter>");
    bn_gb_emit(b, BN_G_JMP, 0, 0, loop, -1);    /* another param or the close */
    int endb = b->n;
    bn_gb_patch(b, sp, -1, endb);
    bn_gb_lit(b, "</function>");
    bn_gb_until_char(b, '<');                   /* WS before </tool_call> */
    bn_gb_lit(b, "</tool_call>");
    return bn_gb_match(b);
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
    int think;               /* enable_thinking, from --thinking on|off */

    /* --grammar: constrain tool-call markup via sampler logit masking. */
    int            grammar_on;
    BnGrammar      grammar;
    BnGInst        grammar_prog[BN_G_MAX_PROG];
    int            grammar_prog_len;
    char         **gvocab_decoded;   /* stable decoded vocab strings (owned) */
    BnGrammarVocab *gvocab;          /* sorted view over gvocab_decoded */
    GrammarMaskCtx  gmctx;
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

/* Decode a token range back to text and echo it (special tokens are literal
 * vocab strings, so the chat-template structure is preserved). */
static void vb_dump_tokens(Agent *a, const char *label,
                           const int *toks, int from, int to) {
    if (!g_verbose) return;
    printf("\n----- %s [%d..%d) -----\n", label, from, to);
    for (int i = from; i < to; i++) {
        const char *p = bn_tokenizer_decode(a->tok, toks[i]);
        if (p) fputs(p, stdout);
    }
    printf("\n");
    fflush(stdout);
}

/* Arm/disarm the tool-call grammar from the committed token stream. Armed when
 * the last <tool_call> in the stream is still open (replaying from that tag
 * recovers the state even if one token straddled the open); advanced per
 * committed piece; disarmed at </tool_call> or if the grammar dies, so free
 * text resumes. Offsets, not pointers, survive the stream buffer reallocating. */
static void grammar_step(Agent *a, ByteBuf *stream, const char *piece,
                         int *armed, size_t *tc_open) {
    const char *buf = stream->buf ? stream->buf : "";
    if (!*armed) {
        const char *open = NULL;
        for (const char *p = buf; (p = strstr(p, "<tool_call>")) != NULL; p += 11)
            open = p;
        if (open && !strstr(open + 11, "</tool_call>")) {
            bn_grammar_init(&a->grammar, a->grammar_prog, a->grammar_prog_len);
            bn_grammar_advance(&a->grammar, open);
            bn_sampler_set_mask(&a->sampler, grammar_mask_cb, &a->gmctx);
            *armed = 1;
            *tc_open = (size_t)(open - buf);
        }
    } else {
        bn_grammar_advance(&a->grammar, piece);
        if (strstr(buf + *tc_open + 11, "</tool_call>") || bn_grammar_dead(&a->grammar)) {
            bn_sampler_set_mask(&a->sampler, NULL, NULL);
            *armed = 0;
        }
    }
}

/* Batch-prefill [tokens..tokens+n) at pos0. The batch already computes the
 * logits for the next token, returned via *out_logits -- the caller MUST use
 * these instead of re-feeding the last token, which would advance the recurrent
 * (SSM) state a second time and corrupt it (hybrid models scan forward; a token
 * applied twice is not idempotent like an attention-KV overwrite). */
static int prefill_range(Agent *a, const int *tokens, int n, int pos0,
                         float **out_logits) {
    if (n <= 0) return 0;
    double t0 = now_ms();
    float *lp = bn_prefill(a->model, a->sess, tokens, n, pos0, 0);
    g_pp_ms += now_ms() - t0;
    g_pp_tok += n;
    if (out_logits) *out_logits = lp;
    return lp ? 0 : -1;
}

static int restore_and_extend(Agent *a, const int *full_tokens, int n_full,
                              float **out_logits) {
    if (bn_session_set_recurrent_state(a->sess, a->model,
                                       a->snap, a->snap_bytes) != 0) {
        fprintf(stderr, "restore failed\n");
        return -1;
    }
    bn_session_kv_truncate(a->sess, a->pos_pre_turn);
    if (n_full > a->pos_pre_turn)
        return prefill_range(a, full_tokens + a->pos_pre_turn,
                             n_full - a->pos_pre_turn, a->pos_pre_turn,
                             out_logits);
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

static int run_turn(Agent *a, const char *user_msg,
                    const BnTplTool *tools[], int n_tools,
                    int turn_index, ByteBuf *transcript) {
    /* per-turn metric accumulators */
    g_pp_ms = g_tg_ms = 0.0;
    g_pp_tok = g_tg_tok = 0;
    /* The sampler persists across turns; start every turn with no mask. */
    if (a->grammar_on) bn_sampler_set_mask(&a->sampler, NULL, NULL);
    double t_turn0 = now_ms();
    double cpu_turn0 = cur_cpu_ms();
    double gpu_turn0 = cur_gpu_active_ms(a->model);
    int m_ctx_in = a->pos_pre_turn;
    int m_tool_ok = 0, m_tool_err = 0;

    int user_idx = a->n_msgs;
    a->msgs[user_idx].role = BN_TPL_ROLE_USER;
    a->msgs[user_idx].content = strdup(user_msg);
    a->msgs[user_idx].reasoning = NULL;
    a->msgs[user_idx].tool_calls = NULL;
    a->msgs[user_idx].n_tool_calls = 0;
    a->n_msgs++;

    int prompt_tokens[MAX_TOKENS];
    int n_prompt = render_msgs(a, a->think, 1,
                               tools, n_tools, prompt_tokens, MAX_TOKENS);
    if (n_prompt < 0) { fprintf(stderr, "render prompt failed\n"); return -1; }
    vb_dump_tokens(a, "PROMPT", prompt_tokens, 0, n_prompt);
    if (g_verbose) { printf("----- GENERATION -----\n"); fflush(stdout); }

    float *logits = NULL;
    if (restore_and_extend(a, prompt_tokens, n_prompt, &logits) != 0) return -1;
    int pos = n_prompt;

    ByteBuf stream = {0};
    size_t scanned_to = 0;
    ParsedToolCall observed_tcs[MAX_TOOL_CALLS_PER_TURN];
    int n_observed_tcs = 0;
    int turn_done = 0;
    int gen_count = 0;
    int round_count = 0;
    int grammar_armed = 0;
    size_t grammar_open = 0;
    char *last_sig = NULL;
    int repeat_rounds = 0;

    if (!logits) { fprintf(stderr, "prompt prefill produced no logits\n"); return -1; }

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
                if (strncmp(tc->result, "error", 5) == 0) m_tool_err++;
                else m_tool_ok++;
                if (g_verbose) {
                    printf("\n[tool] %s(", tc->function_name);
                    for (int p = 0; p < tc->n_params; p++)
                        printf("%s%s=%s", p ? ", " : "",
                               tc->param_names[p], tc->param_values[p]);
                    printf(") -> %s\n", tc->result);
                } else {
                    printf("[tool] %s -> %s\n", tc->function_name, tc->result);
                }
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
            /* Do NOT commit reasoning to history: the chat template strips
               <think> from every assistant turn except the active last one, so a
               stored reasoning makes the re-rendered prefix (and the snapshot
               position it's based on) diverge from the next turn's render. */
            free(reasoning);
            a->msgs[asst_idx].reasoning = NULL;
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

            /* Loop-breaker: a small model can wedge on a tool it cannot satisfy
               (e.g. apply_diff "corrupt patch") and re-emit the same call every
               round until the growing context overflows and prefill crashes.
               End the turn once the same call signature repeats too often. */
            char sig[1024];
            size_t so = 0;
            for (int t = 0; t < n_observed_tcs && so + 1 < sizeof sig; t++) {
                ParsedToolCall *tc = &observed_tcs[t];
                int w = snprintf(sig + so, sizeof sig - so, "%s(", tc->function_name);
                if (w > 0) so += (size_t)w < sizeof sig - so ? (size_t)w : sizeof sig - so - 1;
                for (int p = 0; p < tc->n_params && so + 1 < sizeof sig; p++) {
                    w = snprintf(sig + so, sizeof sig - so, "%s=%s;",
                                 tc->param_names[p], tc->param_values[p]);
                    if (w > 0) so += (size_t)w < sizeof sig - so ? (size_t)w : sizeof sig - so - 1;
                }
            }
            if (last_sig && strcmp(last_sig, sig) == 0) repeat_rounds++;
            else repeat_rounds = 0;
            free(last_sig);
            last_sig = strdup(sig);
            if (repeat_rounds + 1 >= MAX_IDENTICAL_ROUNDS) {
                fprintf(stderr, "loop-breaker: '%s' repeated %d times; ending turn\n",
                        sig, repeat_rounds + 1);
                turn_done = 1;
            }
            /* The repeat check only catches identical CONSECUTIVE calls; an
               alternating A-B-A-B loop evades it, so also cap total rounds. */
            if (round_count >= MAX_TOTAL_ROUNDS) {
                fprintf(stderr, "loop-breaker: %d tool rounds in one turn; ending turn\n",
                        round_count);
                turn_done = 1;
            }

            if (!turn_done) {
                int cont_tokens[MAX_TOKENS];
                int enable_thinking_now = a->think && (round_count < MAX_ROUNDS_PER_TURN);
                int n_cont = render_msgs(a, enable_thinking_now, 1,
                                         tools, n_tools, cont_tokens, MAX_TOKENS);
                if (n_cont < 0) {
                    fprintf(stderr, "render continuation failed\n"); return -1;
                }
                if (n_cont > pos) {
                    vb_dump_tokens(a, "TOOL-RESULT PROMPT", cont_tokens, pos, n_cont);
                    if (g_verbose) { printf("----- GENERATION -----\n"); fflush(stdout); }
                    if (prefill_range(a, cont_tokens + pos, n_cont - pos, pos,
                                      &logits) != 0)
                        return -1;
                    pos = n_cont;
                }
            }
            bb_reset(&stream);
            scanned_to = 0;
            if (a->grammar_on) { bn_sampler_set_mask(&a->sampler, NULL, NULL); grammar_armed = 0; }
            for (int t = 0; t < n_observed_tcs; t++)
                parsed_tc_free(&observed_tcs[t]);
            n_observed_tcs = 0;
            continue;
        }

        const char *piece = bn_tokenizer_decode(a->tok, tid);
        if (piece) bb_append(&stream, piece);
        if (g_verbose && piece) { fputs(piece, stdout); fflush(stdout); }
        if (a->grammar_on && piece)
            grammar_step(a, &stream, piece, &grammar_armed, &grammar_open);

        double t0 = now_ms();
        float *lp = bn_prefill(a->model, a->sess, &tid, 1, pos, 0);
        g_tg_ms += now_ms() - t0;
        g_tg_tok += 1;
        if (!lp) return -1;
        logits = lp;
        pos++;
    }

    if (stream.buf && stream.len > 0) {
        char *reasoning = NULL, *content = NULL;
        extract_reasoning_content(stream.buf, &reasoning, &content);
        /* Thinking is transient; commit only the answer text to history so the
           re-rendered prefix stays stable across turns (see tool-round note). */
        free(reasoning);
        if (content && *content) {
            int asst_idx = a->n_msgs;
            a->msgs[asst_idx].role = BN_TPL_ROLE_ASSISTANT;
            a->msgs[asst_idx].content = content;
            a->msgs[asst_idx].reasoning = NULL;
            a->msgs[asst_idx].tool_calls = NULL;
            a->msgs[asst_idx].n_tool_calls = 0;
            a->n_msgs++;
        } else {
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
    free(last_sig);

    /* Record this turn's metrics before the snapshot re-render below (that
     * prefill is harness bookkeeping, not prompt processing). */
    if (g_n_metrics < MAX_TURNS) {
        TurnMetrics *tm = &g_metrics[g_n_metrics++];
        tm->ctx_in = m_ctx_in;
        tm->ctx_out = pos;
        tm->pp_tok = g_pp_tok;
        tm->pp_ms = g_pp_ms;
        tm->tg_tok = g_tg_tok;
        tm->tg_ms = g_tg_ms;
        tm->rounds = round_count;
        tm->tool_ok = m_tool_ok;
        tm->tool_err = m_tool_err;
        tm->rss = cur_rss_bytes();
        tm->footprint = cur_footprint_bytes();
        tm->wall_ms = now_ms() - t_turn0;
        tm->cpu_ms = cur_cpu_ms() - cpu_turn0;
        tm->gpu_ms = cur_gpu_active_ms(a->model) - gpu_turn0;
        tm->pressure = mem_pressure_level();
    }

    /* Re-render the full conversation (no clean_history concept in bn_chat_template_encode)
       and snapshot so the next turn starts from a consistent KV position. */
    int clean_tokens[MAX_TOKENS];
    int n_clean = render_msgs(a, a->think, 0,
                              tools, n_tools, clean_tokens, MAX_TOKENS);
    if (n_clean < 0) {
        fprintf(stderr, "render clean history failed\n"); return -1;
    }
    if (restore_and_extend(a, clean_tokens, n_clean, NULL) != 0) return -1;
    if (snapshot_current(a) != 0) {
        fprintf(stderr, "snapshot failed\n"); return -1;
    }
    a->pos_pre_turn = n_clean;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <model.gguf> [--out transcript.json] "
                "[--workdir DIR] [--cpu] [--grammar] [--vv] [--kv16] "
                "[--thinking on|off] [--reasoning-effort low|medium|high]\n                "
                "[--maxseq N] [--callfmt 0..5]\n",
                argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *transcript_path = "tr.json";
    const char *workdir = NULL;
    int use_cpu = 0;             /* --cpu: force CPU (Metal is the default) */
    int use_grammar = 0;         /* --grammar: constrain tool-call markup */
    int kv_f16 = 0;   /* --kv16: store the KV cache as fp16 (half the memory) */
    int think_mode = 1;          /* --thinking on|off -> enable_thinking (default on) */
    const char *effort = "low";  /* --reasoning-effort low|medium|high (default low) */
    int max_seq = 8192;          /* --maxseq N: KV/context token cap */
    int callfmt = 0;             /* --callfmt 0..4: tool-call example variant */
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) transcript_path = argv[++i];
        else if (strcmp(argv[i], "--workdir") == 0 && i + 1 < argc) workdir = argv[++i];
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = 1;
        else if (strcmp(argv[i], "--grammar") == 0) use_grammar = 1;
        else if (strcmp(argv[i], "--vv") == 0 || strcmp(argv[i], "--verbose") == 0)
            g_verbose = 1;
        else if (strcmp(argv[i], "--kv16") == 0) kv_f16 = 1;
        else if (strcmp(argv[i], "--thinking") == 0 && i + 1 < argc)
            think_mode = strcmp(argv[++i], "off") != 0;
        else if (strcmp(argv[i], "--reasoning-effort") == 0 && i + 1 < argc)
            effort = argv[++i];
        else if (strcmp(argv[i], "--callfmt") == 0 && i + 1 < argc) {
            callfmt = atoi(argv[++i]);
            if (callfmt < 0) callfmt = 0;
            if (callfmt > 5) callfmt = 5;
        }
        else if (strcmp(argv[i], "--maxseq") == 0 && i + 1 < argc) {
            max_seq = atoi(argv[++i]);
            if (max_seq < 1) max_seq = 1;
            if (max_seq > MAX_TOKENS) max_seq = MAX_TOKENS;
        } else {
            fprintf(stderr, "test_agent: unknown or malformed option '%s'\n\n"
                    "usage: %s <model.gguf> [--out FILE] [--workdir DIR] [--cpu]\n"
                    "  [--grammar] [--vv] [--kv16] [--thinking on|off]\n"
                    "  [--reasoning-effort low|medium|high] [--maxseq N] [--callfmt 0..5]\n",
                    argv[i], argv[0]);
            return 2;
        }
    }

    /* Metal is the default on Apple Silicon (the per-slice arch macro also makes
     * a fat binary / Rosetta run CPU on x64); --cpu forces CPU. */
    int want_metal = 0;
#ifdef BN_ENABLE_METAL
#if defined(__aarch64__) || defined(__arm64__)
    if (!use_cpu) want_metal = 1;
#endif
#endif
    /* If --workdir is set the agent's file tools run inside it (chdir below);
       resolve a relative --out to an absolute path now so the transcript still
       lands in the launch directory, outside the searched sandbox. */
    char transcript_abs[4096];
    if (workdir && transcript_path[0] != '/') {
        char cwd[4096];
        if (getcwd(cwd, sizeof cwd) &&
            snprintf(transcript_abs, sizeof transcript_abs, "%s/%s",
                     cwd, transcript_path) < (int)sizeof transcript_abs)
            transcript_path = transcript_abs;
    }

    BnGGUFFile *gf = bn_gguf_open_file(model_path);
    if (!gf) { fprintf(stderr, "gguf open failed\n"); return 1; }
    BnModel model;
    if (bn_model_load(&model, gf, max_seq, kv_f16, 0) != 0) {
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
       so the main thread and OS stay responsive while the model runs.
       Override with BN_AGENT_THREADS (e.g. =0 lets the engine pick, or set it
       to hw.logicalcpu to saturate all cores). */
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
    const char *env_threads = getenv("BN_AGENT_THREADS");
    if (env_threads) {
        int t = atoi(env_threads);
        if (t >= 0) n_workers = t;
    }
    fprintf(stderr, "CPU thread pool: %d workers\n", n_workers);
    bn_model_set_thread_pool(&model, bn_tp_create(n_workers), 1);

#ifdef BN_ENABLE_METAL
    if (want_metal) {
        BnGPUBackend *gpu = bn_gpu_metal_create("shaders/metal/");
        /* Hand the mmap range to Metal so weight upload wraps the mapped file
           pages zero-copy (newBufferWithBytesNoCopy) instead of allocating a
           second GPU-resident copy. Without this, unified-memory RSS roughly
           doubles (mmap weights + private GPU copy). Mirrors main.c. */
        const BnMappedFile *mf = bn_gguf_primary_file(gf);
        if (gpu && mf && mf->is_mmap && mf->data)
            bn_gpu_metal_set_mmap_range(gpu, mf->data, mf->size);
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
#endif

    /* Run the agent's file tools inside --workdir so its grep/glob/exec never
       see the harness's own transcript or the caller's redirected stdout log
       (which otherwise sit in the searched cwd and can feed a grep match loop).
       Done after Metal init so the relative shader dir resolved first. */
    if (workdir) {
        if (mkdir(workdir, 0700) != 0 && errno != EEXIST) {
            fprintf(stderr, "could not create workdir %s: %s\n",
                    workdir, strerror(errno));
            return 1;
        }
        if (chdir(workdir) != 0) {
            fprintf(stderr, "could not chdir to %s: %s\n",
                    workdir, strerror(errno));
            return 1;
        }
    }

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
    a.think = think_mode;

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

    /* Build the constrained-decoding tables once. The mask matches REAL bytes,
       so the vocab view is over each token's DECODED string -- bn_tokenizer_decode
       returns a shared thread-local buffer, so each is copied to stable storage
       (bn_grammar_vocab_create borrows the pointers for the run's lifetime). */
    a.grammar_on = use_grammar;
    if (a.grammar_on) {
        a.gvocab_decoded = malloc((size_t)tok.vocab_size * sizeof(char *));
        for (int i = 0; i < tok.vocab_size; i++)
            a.gvocab_decoded[i] = strdup(bn_tokenizer_decode(&tok, i));
        a.gvocab = bn_grammar_vocab_create((const char *const *)a.gvocab_decoded,
                                           tok.vocab_size);
        BnGBuilder gb;
        build_tool_call_grammar(&gb);
        memcpy(a.grammar_prog, gb.inst, (size_t)gb.n * sizeof(BnGInst));
        a.grammar_prog_len = gb.n;
        a.gmctx.g = &a.grammar;
        a.gmctx.v = a.gvocab;
    }

    const BnTplTool *tools[8] = {
        tools_get_datetime_def(), tools_read_file_def(),
        tools_file_glob_search_def(), tools_grep_search_def(),
        tools_exec_shell_command_def(), tools_write_file_def(),
        tools_edit_file_def(), tools_apply_diff_def()
    };
    int n_tools = 8;
    a.msgs[0].role = BN_TPL_ROLE_SYSTEM;
    const char *sys_base =
        "You are a coding agent working in the current project directory. "
        "You have file and shell tools: get_datetime, read_file, "
        "file_glob_search, grep_search, exec_shell_command, write_file, "
        "edit_file, apply_diff. Call a tool to inspect or modify files, then "
        "reply with a short final answer after the tool round-trip.";
    /* Tool-call example variants (--callfmt) to study whether demonstrating
       parameter name-matching improves the small model's call syntax. V0 leans
       only on the GGUF template's generic <parameter=example_parameter_1>
       example; V1..V4 append a concrete demonstration. */
    static const char *callfmt_suffix[6] = {
        "",
        /* V1: one concrete worked example */
        "\n\nTo call a tool, follow this exact shape and change only the value "
        "lines, using the chosen function's real parameter names:\n"
        "<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n"
        "</parameter>\n</function>\n</tool_call>",
        /* V2: explicit name-matching rule */
        "\n\nTool-call rule: every <parameter=NAME> must use a NAME copied "
        "verbatim from the chosen function's parameter list above -- read_file: "
        "path; write_file: path, content; grep_search: path, pattern; "
        "exec_shell_command: command; file_glob_search: path; edit_file: path, "
        "changes; apply_diff: diff. The line after <parameter=NAME> is its "
        "value; always close it with </parameter>.",
        /* V3: rule + a two-parameter worked example */
        "\n\nTool-call rule: every <parameter=NAME> must use a NAME copied "
        "verbatim from the chosen function's parameter list above. Example "
        "(write_file takes path and content):\n<tool_call>\n<function=write_file>"
        "\n<parameter=path>\nreport.txt\n</parameter>\n<parameter=content>\n"
        "A one-sentence summary.\n</parameter>\n</function>\n</tool_call>",
        /* V4: exact call for every tool in the set */
        "\n\nExactly how to call each tool (copy the parameter names verbatim):\n"
        "<tool_call><function=get_datetime></function></tool_call>\n"
        "<tool_call><function=read_file><parameter=path>notes.txt</parameter>"
        "</function></tool_call>\n"
        "<tool_call><function=file_glob_search><parameter=path>.</parameter>"
        "</function></tool_call>\n"
        "<tool_call><function=grep_search><parameter=path>.</parameter>"
        "<parameter=pattern>TODO</parameter></function></tool_call>\n"
        "<tool_call><function=exec_shell_command><parameter=command>ls -la"
        "</parameter></function></tool_call>\n"
        "<tool_call><function=write_file><parameter=path>report.txt</parameter>"
        "<parameter=content>A one-sentence summary.</parameter></function></tool_call>\n"
        "<tool_call><function=edit_file><parameter=path>report.txt</parameter>"
        "<parameter=changes>[{\"mode\":\"append\",\"line_start\":-1,"
        "\"line_end\":-1,\"content\":\"reviewed: yes\"}]</parameter>"
        "</function></tool_call>\n"
        "<tool_call><function=apply_diff><parameter=diff>--- a/report.txt\n"
        "+++ b/report.txt\n@@ -1 +1 @@\n-reviewed: yes\n+reviewed: done\n"
        "</parameter></function></tool_call>",
        /* V5: V4 (exact per-tool calls) + a contrastive NEGATIVE for the most
           common structural malformation -- the HTML-attribute collapse and
           bare/mismatched tags the judge flagged. */
        "\n\nExactly how to call each tool (copy the parameter names verbatim):\n"
        "<tool_call><function=get_datetime></function></tool_call>\n"
        "<tool_call><function=read_file><parameter=path>notes.txt</parameter>"
        "</function></tool_call>\n"
        "<tool_call><function=grep_search><parameter=path>.</parameter>"
        "<parameter=pattern>TODO</parameter></function></tool_call>\n"
        "<tool_call><function=exec_shell_command><parameter=command>ls -la"
        "</parameter></function></tool_call>\n"
        "<tool_call><function=write_file><parameter=path>report.txt</parameter>"
        "<parameter=content>A one-sentence summary.</parameter></function></tool_call>\n"
        "<tool_call><function=file_glob_search><parameter=path>.</parameter>"
        "</function></tool_call>\n"
        "<tool_call><function=edit_file><parameter=path>report.txt</parameter>"
        "<parameter=changes>[{\"mode\":\"append\",\"line_start\":-1,"
        "\"line_end\":-1,\"content\":\"reviewed: yes\"}]</parameter>"
        "</function></tool_call>\n"
        "<tool_call><function=apply_diff><parameter=diff>--- a/report.txt\n"
        "+++ b/report.txt\n@@ -1 +1 @@\n-reviewed: yes\n+reviewed: done\n"
        "</parameter></function></tool_call>\n"
        "NEVER use attribute style: WRONG `<parameter=pattern>\"TODO\">` and "
        "WRONG `<pattern=...>`. ALWAYS open `<parameter=NAME>`, put the value on "
        "its own line, then close `</parameter>` on its own line. The tag name is "
        "always literally 'parameter', never the parameter's name.",
    };
    if (callfmt > 5) callfmt = 5;
    const char *cf = callfmt_suffix[callfmt];
    /* Qwen3.5's chat template has no reasoning_effort hook, so steer effort via
       a system directive instead of a template variable. */
    const char *steer = !effort ? "" :
        strcmp(effort, "low") == 0 ?
            "Keep reasoning minimal and act quickly." :
        strcmp(effort, "high") == 0 ?
            "Reason carefully and thoroughly before acting." :
            "Use a moderate amount of reasoning.";
    size_t cap = strlen(sys_base) + (effort ? strlen(effort) : 0) +
                 strlen(steer) + strlen(cf) + 48;
    char *sys = malloc(cap);
    if (effort)
        snprintf(sys, cap, "%s\n\nReasoning effort: %s. %s%s",
                 sys_base, effort, steer, cf);
    else
        snprintf(sys, cap, "%s%s", sys_base, cf);
    a.msgs[0].content = sys;
    a.n_msgs = 1;
    printf("config: thinking=%s reasoning-effort=%s callfmt=%d maxseq=%d "
           "kv16=%d metal=%d grammar=%d\n",
           think_mode ? "on" : "off", effort ? effort : "(none)",
           callfmt, max_seq, kv_f16, want_metal, use_grammar);

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

    print_metrics_table(model_name_buf);

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
    if (a.grammar_on) {
        bn_grammar_vocab_destroy(a.gvocab);
        for (int i = 0; i < tok.vocab_size; i++) free(a.gvocab_decoded[i]);
        free(a.gvocab_decoded);
    }
    free(a.snap);
    bn_session_free(a.sess, NULL);
    bn_tokenizer_free(&tok);
    bn_model_free(&model);
    bn_gguf_free(gf);
    return 0;
}
