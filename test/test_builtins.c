/* Engine builtins that have no stock-Jinja2 equivalent, so they cannot be
 * compared against the Jinja2 golden oracle. strftime_now reads the wall
 * clock; we pin it by rendering and comparing against a direct strftime()
 * of the same instant in this process. */
#include "jinja.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static struct jinja_value jv_undef_(void) {
    struct jinja_value v;
    memset(&v, 0, sizeof v);
    v.kind = JV_UNDEFINED;
    return v;
}

static struct jinja_value h_get(void *c, struct jinja_value o,
                                const char *n) {
    (void)c; (void)o; (void)n;
    return jv_undef_();
}

static struct jinja_value h_index(void *c, struct jinja_value o, long i) {
    (void)c; (void)o; (void)i;
    return jv_undef_();
}

static long h_len(void *c, struct jinja_value o) {
    (void)c; (void)o;
    return 0;
}

static int h_truthy(void *c, struct jinja_value v) {
    (void)c; (void)v;
    return 0;
}

static int h_test(void *c, struct jinja_value v, const char *n) {
    (void)c; (void)v; (void)n;
    return 0;
}

static struct jinja_value h_method(void *c, struct jinja_value o,
                                   const char *n, struct jinja_value a) {
    (void)c; (void)o; (void)n; (void)a;
    return jv_undef_();
}

static int fails = 0;

/* Render `tmpl` and require it equals strftime(fmt) of the current time.
 * Sampling the clock immediately before render keeps the two within the
 * same second except across a rollover, which the second sample absorbs. */
static void check(const struct jinja_host *h, const char *name,
                  const char *tmpl, const char *fmt) {
    char want[256];
    time_t t0 = time(NULL);
    strftime(want, sizeof want, fmt, localtime(&t0));
    char *got = jinja_render(tmpl, h, NULL, 0);
    int ok = got && !strcmp(got, want);
    if (!ok) {
        char alt[256];
        time_t t1 = time(NULL);
        strftime(alt, sizeof alt, fmt, localtime(&t1));
        ok = got && !strcmp(got, alt);
    }
    printf("%-22s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) {
        fails++;
        printf("  want %s got %s\n", want, got ? got : "NULL");
    }
    jinja_free(got);
}

/* Render `tmpl` and require an exact match. Used for the builtin-global
 * `is defined` results, which are deterministic but disagree with stock
 * Jinja2 (it lacks strftime_now/raise_exception), so they cannot be
 * golden cases; the authority here is llama.cpp/transformers. */
static void eq(const struct jinja_host *h, const char *name,
               const char *tmpl, const char *want) {
    char *got = jinja_render(tmpl, h, NULL, 0);
    int ok = got && !strcmp(got, want);
    printf("%-22s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) {
        fails++;
        printf("  want %s got %s\n", want, got ? got : "NULL");
    }
    jinja_free(got);
}

int main(void) {
    struct jinja_host h;
    h.get = h_get;
    h.index = h_index;
    h.len = h_len;
    h.truthy = h_truthy;
    h.test = h_test;
    h.method = h_method;
    h.ctx = NULL;
    check(&h, "strftime_now_iso", "{{ strftime_now('%Y-%m-%d') }}",
          "%Y-%m-%d");
    check(&h, "strftime_now_words", "{{ strftime_now('%d %B %Y') }}",
          "%d %B %Y");
    check(&h, "strftime_now_default", "{{ strftime_now() }}", "%Y-%m-%d");
    eq(&h, "strftime_now_defined", "{{ strftime_now is defined }}", "True");
    eq(&h, "raise_exc_defined", "{{ raise_exception is defined }}",
       "True");
    eq(&h, "missing_defined", "{{ nope_xyz is defined }}", "False");
    eq(&h, "llama_date_guard",
       "{%- if strftime_now is defined -%}live{%- else -%}"
       "stale{%- endif -%}", "live");
    printf(fails ? "BUILTIN TESTS FAILED\n" : "ALL BUILTIN TESTS PASS\n");
    return fails ? 1 : 0;
}
