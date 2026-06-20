#include "grammar.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* next < 0 means "fall through to idx + 1". */
static int resolve_next(const BnGInst *p, int idx) {
    return p[idx].next < 0 ? idx + 1 : p[idx].next;
}

static int is_consumer(BnGOp op) {
    return op == BN_G_CHAR || op == BN_G_RANGE ||
           op == BN_G_ANY  || op == BN_G_NOT;
}

/* Epsilon-closure of the seed indices: follow SPLIT/JMP, collect the
 * char-consuming and MATCH instructions reachable without consuming a byte.
 * The "seen" set uses a generation counter instead of a per-call memset --
 * closure is the inner loop of vocab masking (called ~once per trie edge), so
 * a 256-byte clear per call would dominate. Single-threaded (sampling) path. */
static void closure(const BnGInst *p, int len, const int *seeds, int n_seeds,
                    int *out, int *n_out) {
    static int seen[BN_G_MAX_PROG];
    static unsigned gen;
    if (++gen == 0) { memset(seen, 0, sizeof seen); gen = 1; }
    int g = (int)gen;
    int stack[BN_G_MAX_ACTIVE];
    int sp = 0;
    for (int i = 0; i < n_seeds; i++) {
        int idx = seeds[i];
        if (idx >= 0 && idx < len && seen[idx] != g) { seen[idx] = g; stack[sp++] = idx; }
    }
    int n = 0;
    while (sp > 0) {
        int idx = stack[--sp];
        BnGOp op = (BnGOp)p[idx].op;
        if (op == BN_G_SPLIT) {
            int a = resolve_next(p, idx);
            int b = p[idx].alt;
            if (a >= 0 && a < len && seen[a] != g) { seen[a] = g; stack[sp++] = a; }
            if (b >= 0 && b < len && seen[b] != g) { seen[b] = g; stack[sp++] = b; }
        } else if (op == BN_G_JMP) {
            int a = resolve_next(p, idx);
            if (a >= 0 && a < len && seen[a] != g) { seen[a] = g; stack[sp++] = a; }
        } else if ((is_consumer(op) || op == BN_G_MATCH) && n < BN_G_MAX_ACTIVE) {
            out[n++] = idx;
        }
    }
    *n_out = n;
}

void bn_grammar_init(BnGrammar *g, const BnGInst *prog, int prog_len) {
    g->prog = prog;
    g->prog_len = prog_len;
    int seed = 0;
    closure(prog, prog_len, &seed, 1, g->active, &g->n_active);
}

static int set_has_match(const BnGInst *p, const int *set, int n) {
    for (int i = 0; i < n; i++)
        if (p[set[i]].op == BN_G_MATCH) return 1;
    return 0;
}

static int match_byte(const BnGInst *in, unsigned char c) {
    switch ((BnGOp)in->op) {
        case BN_G_CHAR:  return c == in->a;
        case BN_G_RANGE: return c >= in->a && c <= in->b;
        case BN_G_ANY:   return c != 0;
        case BN_G_NOT:   return c != in->a;
        default:         return 0;
    }
}

/* Step the active set over byte c into out; returns out size. A MATCH state is
 * absorbing: it accepts and consumes any trailing byte, so once the grammar is
 * satisfied the remainder of the text (and the token) is unconstrained -- and
 * the masking state-stack stays complete past the grammar's end. */
static int step(const BnGInst *p, int len, const int *cur, int n_cur,
                unsigned char c, int *out) {
    int seeds[BN_G_MAX_ACTIVE];
    int ns = 0;
    int match_idx = -1;
    for (int i = 0; i < n_cur; i++) {
        BnGOp op = (BnGOp)p[cur[i]].op;
        if (op == BN_G_MATCH) {
            match_idx = cur[i];
        } else if (is_consumer(op) && match_byte(&p[cur[i]], c) &&
                   ns < BN_G_MAX_ACTIVE) {
            seeds[ns++] = resolve_next(p, cur[i]);
        }
    }
    int n_out = 0;
    closure(p, len, seeds, ns, out, &n_out);
    if (match_idx >= 0) {
        int present = 0;
        for (int j = 0; j < n_out; j++)
            if (out[j] == match_idx) { present = 1; break; }
        if (!present && n_out < BN_G_MAX_ACTIVE) out[n_out++] = match_idx;
    }
    return n_out;
}

/* Simulate s from the current active set. If keep != NULL, write the final
 * active set there (commit). Returns true if the token stays grammar-legal. */
static bool simulate(const BnGrammar *g, const char *s, int *keep, int *n_keep) {
    int cur[BN_G_MAX_ACTIVE], nxt[BN_G_MAX_ACTIVE];
    int n = g->n_active;
    memcpy(cur, g->active, (size_t)n * sizeof(int));
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        int nn = step(g->prog, g->prog_len, cur, n, *p, nxt);
        if (nn == 0) return false;
        memcpy(cur, nxt, (size_t)nn * sizeof(int));
        n = nn;
    }
    if (keep) { memcpy(keep, cur, (size_t)n * sizeof(int)); *n_keep = n; }
    return n > 0;
}

bool bn_grammar_accepts(const BnGrammar *g, const char *s) {
    if (!s || !s[0]) return false;
    return simulate(g, s, NULL, NULL);
}

void bn_grammar_advance(BnGrammar *g, const char *s) {
    int keep[BN_G_MAX_ACTIVE], nk = 0;
    if (simulate(g, s, keep, &nk)) {
        memcpy(g->active, keep, (size_t)nk * sizeof(int));
        g->n_active = nk;
    } else {
        g->n_active = 0;
    }
}

bool bn_grammar_can_stop(const BnGrammar *g) {
    return set_has_match(g->prog, g->active, g->n_active);
}

bool bn_grammar_dead(const BnGrammar *g) {
    return g->n_active == 0;
}

/* -------------------------- builder -------------------------------------- */

void bn_gb_reset(BnGBuilder *b) { b->n = 0; }

int bn_gb_emit(BnGBuilder *b, BnGOp op, uint8_t a, uint8_t bb,
               int next, int alt) {
    if (b->n >= BN_G_MAX_PROG) return -1;
    int i = b->n++;
    b->inst[i].op = (uint8_t)op;
    b->inst[i].a = a;
    b->inst[i].b = bb;
    b->inst[i].next = (int16_t)next;
    b->inst[i].alt = (int16_t)alt;
    return i;
}

int bn_gb_lit(BnGBuilder *b, const char *s) {
    int first = b->n;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++)
        bn_gb_emit(b, BN_G_CHAR, *p, 0, -1, -1);
    return first;
}

int bn_gb_until_char(BnGBuilder *b, uint8_t c) {
    int i = b->n;
    bn_gb_emit(b, BN_G_SPLIT, 0, 0, i + 1, i + 2);  /* loop body | exit     */
    bn_gb_emit(b, BN_G_NOT,   c, 0, i,     -1);     /* any!=c, back to split */
    return i;
}

int bn_gb_match(BnGBuilder *b) {
    return bn_gb_emit(b, BN_G_MATCH, 0, 0, -1, -1);
}

int bn_gb_ws_star(BnGBuilder *b) {
    int loop = bn_gb_emit(b, BN_G_SPLIT, 0, 0, -1, 0);  /* body | exit (alt patch) */
    int a1   = bn_gb_emit(b, BN_G_SPLIT, 0, 0, -1, 0);   /* ' '  | next */
    bn_gb_emit(b, BN_G_CHAR, ' ',  0, loop, -1);
    int a2   = bn_gb_emit(b, BN_G_SPLIT, 0, 0, -1, 0);   /* '\t' | next */
    bn_gb_emit(b, BN_G_CHAR, '\t', 0, loop, -1);
    int a3   = bn_gb_emit(b, BN_G_SPLIT, 0, 0, -1, 0);   /* '\n' | '\r' */
    bn_gb_emit(b, BN_G_CHAR, '\n', 0, loop, -1);
    bn_gb_emit(b, BN_G_CHAR, '\r', 0, loop, -1);
    int after = b->n;
    bn_gb_patch(b, loop, -1, after);
    bn_gb_patch(b, a1,   -1, a1 + 2);
    bn_gb_patch(b, a2,   -1, a2 + 2);
    bn_gb_patch(b, a3,   -1, a3 + 2);
    return loop;
}

void bn_gb_patch(BnGBuilder *b, int idx, int next, int alt) {
    if (idx < 0 || idx >= b->n) return;
    if (next >= 0) b->inst[idx].next = (int16_t)next;
    if (alt  >= 0) b->inst[idx].alt  = (int16_t)alt;
}

/* <tool_call> WS <function=NAME> ( WS <parameter=PNAME> VALUE </parameter> )* WS
 * </function> WS </tool_call>. WS = (any byte except '<')*, absorbing inter-tag
 * newlines and matching the zero-space inline form. Parameters are zero-or-more
 * so a no-arg call emits </function> straight after <function=NAME>. Names run
 * until '>', values until '<' (a value may contain '>' but not '<'). */
int bn_grammar_build_tool_call(BnGBuilder *b) {
    bn_gb_reset(b);
    bn_gb_lit(b, "<tool_call>");
    bn_gb_ws_star(b);                           /* WS before <function= */
    bn_gb_lit(b, "<function=");
    bn_gb_until_char(b, '>');                   /* function name, free */
    bn_gb_lit(b, ">");
    int loop = bn_gb_ws_star(b);                /* WS before the next tag */
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
    bn_gb_ws_star(b);                           /* WS before </tool_call> */
    bn_gb_lit(b, "</tool_call>");
    return bn_gb_match(b);
}

/* ---------------- fast vocab masking (sorted prefix walk) ---------------- */

struct BnGrammarVocab {
    int n;
    const char **s;   /* sorted by string (borrowed pointers)              */
    int *id;          /* token id for each sorted entry                    */
    int *len;         /* byte length of each                               */
    int max_len;
    int *st;          /* state stack: (max_len+1) * BN_G_MAX_ACTIVE        */
    int *stn;         /* per-depth active count: (max_len+1)               */
};

typedef struct { const char *s; int id; } VPair;

static int vpair_cmp(const void *a, const void *b) {
    return strcmp(((const VPair *)a)->s, ((const VPair *)b)->s);
}

BnGrammarVocab *bn_grammar_vocab_create(const char *const *strs, int n) {
    if (!strs || n <= 0) return NULL;
    BnGrammarVocab *v = (BnGrammarVocab *)calloc(1, sizeof *v);
    VPair *pairs = (VPair *)malloc((size_t)n * sizeof *pairs);
    if (!v || !pairs) { free(v); free(pairs); return NULL; }
    int max_len = 1;
    for (int i = 0; i < n; i++) {
        pairs[i].s = strs[i] ? strs[i] : "";
        pairs[i].id = i;
        int l = (int)strlen(pairs[i].s);
        if (l > max_len) max_len = l;
    }
    qsort(pairs, (size_t)n, sizeof *pairs, vpair_cmp);
    v->n = n;
    v->max_len = max_len;
    v->s   = (const char **)malloc((size_t)n * sizeof *v->s);
    v->id  = (int *)malloc((size_t)n * sizeof *v->id);
    v->len = (int *)malloc((size_t)n * sizeof *v->len);
    v->st  = (int *)malloc((size_t)(max_len + 1) * BN_G_MAX_ACTIVE * sizeof *v->st);
    v->stn = (int *)malloc((size_t)(max_len + 1) * sizeof *v->stn);
    if (!v->s || !v->id || !v->len || !v->st || !v->stn) {
        bn_grammar_vocab_destroy(v); free(pairs); return NULL;
    }
    for (int i = 0; i < n; i++) {
        v->s[i] = pairs[i].s;
        v->id[i] = pairs[i].id;
        v->len[i] = (int)strlen(pairs[i].s);
    }
    free(pairs);
    return v;
}

void bn_grammar_vocab_destroy(BnGrammarVocab *v) {
    if (!v) return;
    free(v->s); free(v->id); free(v->len); free(v->st); free(v->stn);
    free(v);
}

void bn_grammar_mask_logits(const BnGrammar *g, const BnGrammarVocab *v,
                            float *logits) {
    const int A = BN_G_MAX_ACTIVE;
    int *st = v->st;
    int *stn = v->stn;
    int nseed = g->n_active < A ? g->n_active : A;
    memcpy(st, g->active, (size_t)nseed * sizeof(int));
    stn[0] = nseed;
    const char *prev = "";
    int prev_len = 0;
    int valid = 0;  /* stn[0..valid] are correct for the current stack contents */
    for (int k = 0; k < v->n; k++) {
        const char *s = v->s[k];
        int len = v->len[k];
        if (len > v->max_len) continue;
        int lcp = 0;
        while (lcp < prev_len && lcp < len && prev[lcp] == s[lcp]) lcp++;
        int d = lcp < valid ? lcp : valid;  /* reuse min(lcp, valid) depths   */
        while (d < len && stn[d] > 0) {
            stn[d + 1] = step(g->prog, g->prog_len, &st[d * A], stn[d],
                              (unsigned char)s[d], &st[(d + 1) * A]);
            d++;
        }
        if (!((d == len) && stn[len] > 0))
            logits[v->id[k]] = -INFINITY;
        valid = d;
        prev = s;
        prev_len = len;
    }
}
