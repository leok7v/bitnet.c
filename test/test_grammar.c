/* Unit tests for the regexp-NFA grammar engine (grammar.h). Standalone:
 *   cc -Iinclude test/test_grammar.c src/grammar.c -o test_grammar && ./test_grammar
 * Exercises literal runs, until-char free values, alternation/loops, MATCH
 * stop semantics, and the multi-char-token boundary problem that motivated
 * the design (one token spanning several rules). */
#include "grammar.h"
#include <stdio.h>
#include <string.h>

static int g_fail;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); g_fail++; } \
} while (0)

/* Build {"age":<digits>} -- Leo's GBNG.txt example. */
static int build_json(BnGBuilder *b) {
    bn_gb_reset(b);
    bn_gb_lit(b, "{\"age\":");
    int loop = b->n;                         /* one-or-more digits          */
    bn_gb_emit(b, BN_G_RANGE, '0', '9', -1, -1);   /* first digit (i)       */
    bn_gb_emit(b, BN_G_SPLIT, 0, 0, loop, b->n + 1);/* more digits | exit   */
    bn_gb_lit(b, "}");
    return bn_gb_match(b);
}

/* Build a one-param tool-call skeleton with the loop/alternation and free
 * (until '<') value, the shape the agent needs:
 *   <c><parameter= NAME > VALUE </parameter> ( <parameter=... | </c> ) ... */
static int build_call(BnGBuilder *b) {
    bn_gb_reset(b);
    int param = b->n;
    bn_gb_lit(b, "<parameter=");
    bn_gb_until_char(b, '>');                /* param name, free            */
    bn_gb_lit(b, ">");
    bn_gb_until_char(b, '<');                /* value, free until close     */
    bn_gb_lit(b, "</parameter>");
    int after = b->n;
    /* loop: another <parameter= OR the closing </c> */
    int sp = bn_gb_emit(b, BN_G_SPLIT, 0, 0, param, b->n + 1);
    (void)after; (void)sp;
    bn_gb_lit(b, "</c>");
    return bn_gb_match(b);
}

int main(void) {
    BnGBuilder b;
    BnGrammar g;

    /* ---- JSON ---- */
    build_json(&b);
    bn_grammar_init(&g, b.inst, b.n);
    CHECK(bn_grammar_accepts(&g, "{\""), "json: accept opening");
    CHECK(!bn_grammar_accepts(&g, "x"), "json: reject non-'{'");
    bn_grammar_advance(&g, "{\"age\":");     /* multi-char token spanning rules */
    CHECK(!bn_grammar_dead(&g), "json: alive after key");
    CHECK(bn_grammar_accepts(&g, "5"), "json: accept digit");
    CHECK(!bn_grammar_accepts(&g, "}"), "json: '+' needs >=1 digit before '}'");
    bn_grammar_advance(&g, "5");
    CHECK(bn_grammar_accepts(&g, "}"), "json: accept '}' after a digit");
    CHECK(bn_grammar_accepts(&g, "42}"), "json: accept 'more digits + }'");
    bn_grammar_advance(&g, "}");
    CHECK(bn_grammar_can_stop(&g), "json: MATCH after '}'");

    /* ---- tool-call skeleton: structure is forced, name/value are free ---- */
    build_call(&b);
    bn_grammar_init(&g, b.inst, b.n);
    /* the literal '<parameter=' is forced: a bad tag is masked */
    CHECK(bn_grammar_accepts(&g, "<parameter=path>"), "call: accept good open+name");
    CHECK(!bn_grammar_accepts(&g, "<pattern="), "call: reject <pattern= (wrong tag)");
    CHECK(!bn_grammar_accepts(&g, "<p=\"x\">"), "call: reject attribute-collapse");
    bn_grammar_advance(&g, "<parameter=");
    CHECK(bn_grammar_accepts(&g, "path"), "call: free param name");
    CHECK(bn_grammar_accepts(&g, "anything_here"), "call: any name byte ok");
    bn_grammar_advance(&g, "path>");         /* name + '>' */
    bn_grammar_advance(&g, "notes.txt");     /* free value */
    CHECK(bn_grammar_accepts(&g, "</parameter>"), "call: accept close after value");
    CHECK(!bn_grammar_accepts(&g, "junk</wrong>"), "call: value must close with </parameter>");
    bn_grammar_advance(&g, "</parameter>");
    /* loop point: BOTH another <parameter= and the closing </c> are legal */
    CHECK(bn_grammar_accepts(&g, "<parameter=q>"), "call: accept a second param");
    CHECK(bn_grammar_accepts(&g, "</c>"), "call: accept end-of-call");
    CHECK(!bn_grammar_accepts(&g, "garbage"), "call: reject junk at loop point");
    bn_grammar_advance(&g, "</c>");
    CHECK(bn_grammar_can_stop(&g), "call: MATCH after </c>");

    /* ---- fast vocab masking: same shape as the sampler hot path ---- */
    static const char *vocab[] = {
        "<parameter=", "<pattern=", "<p=", "<", "<pa", "path", ">",
        "foo", "</c>", "x", "</parameter>", "abc",
    };
    enum { V_PARAM, V_PATTERN, V_PEQ, V_LT, V_LTPA, V_PATH, V_GT,
           V_FOO, V_ENDC, V_X, V_CLOSE, V_ABC, V_N };
    BnGrammarVocab *gv = bn_grammar_vocab_create(vocab, V_N);
    CHECK(gv != NULL, "mask: vocab built");
    float logit[V_N];

    /* State: grammar expects '<parameter=' (start of build_call). */
    build_call(&b);
    bn_grammar_init(&g, b.inst, b.n);
    for (int i = 0; i < V_N; i++) logit[i] = 0.0f;
    bn_grammar_mask_logits(&g, gv, logit);
    CHECK(logit[V_PARAM] == 0.0f, "mask: '<parameter=' allowed");
    CHECK(logit[V_LT]    == 0.0f, "mask: '<' (valid prefix) allowed");
    CHECK(logit[V_LTPA]  == 0.0f, "mask: '<pa' (valid prefix) allowed");
    CHECK(logit[V_PATTERN] < 0, "mask: '<pattern=' forbidden");
    CHECK(logit[V_PEQ]     < 0, "mask: '<p=' forbidden");
    CHECK(logit[V_PATH]    < 0, "mask: 'path' forbidden at tag position");
    CHECK(logit[V_GT]      < 0, "mask: '>' forbidden at tag position");
    CHECK(logit[V_FOO]     < 0, "mask: 'foo' forbidden at tag position");

    /* Advance into a value (free run until '</parameter>'). */
    bn_grammar_advance(&g, "<parameter=path>");
    for (int i = 0; i < V_N; i++) logit[i] = 0.0f;
    bn_grammar_mask_logits(&g, gv, logit);
    CHECK(logit[V_FOO]   == 0.0f, "mask: value byte 'foo' allowed");
    CHECK(logit[V_GT]    == 0.0f, "mask: value byte '>' allowed");
    CHECK(logit[V_CLOSE] == 0.0f, "mask: '</parameter>' close allowed");
    CHECK(logit[V_PARAM] < 0, "mask: '<parameter=' forbidden inside value");

    bn_grammar_vocab_destroy(gv);

    if (g_fail == 0) printf("All grammar tests passed!\n");
    else printf("%d grammar test(s) FAILED\n", g_fail);
    return g_fail ? 1 : 0;
}
