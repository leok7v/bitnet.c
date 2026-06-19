#include "jinja.h"

#include <ctype.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { TK_END, TK_ID, TK_STR, TK_INT, TK_OP };
enum {
    ST_NONE, ST_ELIF, ST_ELSE, ST_ENDIF, ST_ENDFOR, ST_ENDMACRO,
    ST_ENDSET, ST_ENDFILTER, ST_EOF,
};

#define DICT_TAG (-31337)
#define MAX_MACROS 16
#define MAX_PARAMS 16
#define MAX_ARGS 16
/* Backstop against runaway recursion: self/mutually-recursive macros whose
 * termination condition the engine fails to bind (e.g. an unsupported
 * feature drops the flag that would stop them). Hitting this raises and
 * jinja_render returns NULL instead of overflowing the C stack. Legitimate
 * chat-template nesting is well under this. */
#define MAX_DEPTH 512

struct arena_blk {
    struct arena_blk *next;
    char data[1];
};

struct outbuf {
    char *buf;
    size_t len;
    size_t cap;
};

struct list {
    struct jinja_value *items;
    int n;
};

struct dict {
    struct jinja_var *e;
    int n;
    int cap;
};

struct macro {
    const char *name;
    int nlen;
    struct {
        const char *name;
        int nlen;
        const char *def;
    } param[MAX_PARAMS];
    int nparam;
    const char *body;
};

struct callargs {
    struct jinja_value pos[MAX_ARGS];
    int npos;
    struct {
        const char *name;
        int nlen;
        struct jinja_value val;
    } kw[MAX_ARGS];
    int nkw;
};

struct state {
    const char *src;
    const char *p;
    const struct jinja_host *host;
    struct jinja_var *vars;
    int n_vars;
    int cap_vars;
    struct arena_blk *arena;
    struct outbuf out;
    struct macro macros[MAX_MACROS];
    int n_macros;
    int mute;
    int trim_left;
    int depth;
    int brace;
    int kind;
    const char *ts;
    int tn;
    long iv;
    const char *err;
    jmp_buf jmp;
};

/* All allocation runs through one checker. On a 64-bit overcommit host
 * malloc rarely returns an actionable NULL (the OOM-killer fires first), so
 * the default on failure is abort(). jinja_oom is NULL by loader rules; a
 * caller wanting soft failure assigns a handler that longjmps or exits. */
void (*jinja_oom)(void);

static void *oom(void *p) {
    if (!p) {
        if (jinja_oom) { jinja_oom(); }
        abort();
    }
    return p;
}

static void *xalloc(size_t n) {
    return oom(malloc(n));
}

static void *xresize(void *p, size_t n) {
    return oom(realloc(p, n));
}

static void *arena_alloc(struct state *st, size_t n) {
    struct arena_blk *b =
        (struct arena_blk *)xalloc(sizeof(struct arena_blk) + n);
    b->next = st->arena;
    st->arena = b;
    return b->data;
}

static const char *arena_dup(struct state *st, const char *s, size_t n) {
    char *r = (char *)arena_alloc(st, n + 1);
    memcpy(r, s, n);
    r[n] = '\0';
    return r;
}

/* raise_exception and unsupported constructs unwind here; jinja_render
 * catches it and returns NULL. */
static void fail(struct state *st, const char *msg) {
    st->err = msg;
    longjmp(st->jmp, 1);
}

/* The single chokepoint for runtime-resolved constructs the engine does
 * not implement (filters, tests, functions, methods). In a dead (muted)
 * branch it is a no-op: Jinja resolves these at run time, so one that never
 * executes is not an error -- failing here would reject templates Jinja
 * renders (e.g. a strftime_now call in a not-taken else). On a live path it
 * logs a breadcrumb and fails (NULL): a silently-wrong prompt (an unknown
 * filter no-op'd to identity) is more dangerous than an absent one. The
 * only safe axis is how loudly we refuse, not whether. NOTE: unknown *tags*
 * do NOT come through here -- Jinja rejects them at parse time even in dead
 * branches, so render_block fails on them unconditionally. A future
 * `strictness` field could add an audit mode that collects every live-path
 * gap in one pass and still returns NULL, but never a render-anyway mode.
 * Add the missing feature here rather than relaxing this. */
static void unsupported(struct state *st, const char *what,
                        const char *name, int n) {
    if (st->mute) {
        return;
    }
    fprintf(stderr, "jinja: unsupported %s: %.*s\n", what, n, name);
    fail(st, what);
}

static void out_putn(struct state *st, const char *s, size_t n) {
    if (!st->mute && n) {
        size_t need = st->out.len + n + 1;
        if (need > st->out.cap) {
            size_t nc = st->out.cap ? st->out.cap : 256;
            while (nc < need) {
                nc *= 2;
            }
            st->out.buf = (char *)xresize(st->out.buf, nc);
            st->out.cap = nc;
        }
        memcpy(st->out.buf + st->out.len, s, n);
        st->out.len += n;
        st->out.buf[st->out.len] = '\0';
    }
}

static struct jinja_value jv_none(void) {
    struct jinja_value v = {0};
    v.kind = JV_NONE;
    return v;
}

static struct jinja_value jv_undef(void) {
    struct jinja_value v = {0};
    v.kind = JV_UNDEFINED;
    return v;
}

static struct jinja_value jv_bool(int b) {
    struct jinja_value v = {0};
    v.kind = JV_BOOL;
    v.i = !!b;
    return v;
}

static struct jinja_value jv_int(long n) {
    struct jinja_value v = {0};
    v.kind = JV_INT;
    v.i = n;
    return v;
}

static struct jinja_value jv_str(const char *s) {
    struct jinja_value v = {0};
    v.kind = JV_STR;
    v.s = s ? s : "";
    return v;
}

static struct jinja_value jv_listv(struct list *l) {
    struct jinja_value v = {0};
    v.kind = JV_LIST;
    v.node = l;
    return v;
}

static struct list *list_make(struct state *st, int n) {
    struct list *l = (struct list *)arena_alloc(st, sizeof(struct list));
    l->items = (struct jinja_value *)arena_alloc(st,
                       (size_t)(n ? n : 1) * sizeof(struct jinja_value));
    l->n = 0;
    return l;
}

static int truthy(struct state *st, struct jinja_value v) {
    int r = 0;
    if (v.kind == JV_BOOL || v.kind == JV_INT) {
        r = v.i != 0;
    } else if (v.kind == JV_STR) {
        r = v.s[0] != '\0';
    } else if (v.kind == JV_LIST) {
        r = ((const struct list *)v.node)->n != 0;
    } else if (v.kind == JV_NODE) {
        r = st->host->truthy(st->host->ctx, v);
    }
    return r;
}

/* Render a value to text for output, concat, and the `string` filter. */
static const char *to_str(struct state *st, struct jinja_value v) {
    const char *r = "";
    char buf[32];
    if (v.kind == JV_STR) {
        r = v.s;
    } else if (v.kind == JV_INT) {
        sprintf(buf, "%ld", v.i);
        r = arena_dup(st, buf, strlen(buf));
    } else if (v.kind == JV_BOOL) {
        r = v.i ? "True" : "False";
    } else if (v.kind == JV_NONE) {
        r = "None";
    }
    return r;
}

static const char *as_str(struct jinja_value v) {
    return v.kind == JV_STR ? v.s : "";
}

static int jv_eq(struct jinja_value a, struct jinja_value b) {
    int r = 0;
    if (a.kind == JV_STR || b.kind == JV_STR) {
        r = a.kind == JV_STR && b.kind == JV_STR && strcmp(a.s, b.s) == 0;
    } else if (a.kind == JV_NONE || a.kind == JV_UNDEFINED ||
               b.kind == JV_NONE || b.kind == JV_UNDEFINED) {
        r = a.kind == b.kind;
    } else {
        r = a.i == b.i;
    }
    return r;
}

static struct jinja_value *var_find(struct state *st, const char *name,
                                    int n) {
    struct jinja_value *r = NULL;
    int i = 0;
    while (i < st->n_vars && !r) {
        if ((int)strlen(st->vars[i].name) == n &&
            memcmp(st->vars[i].name, name, (size_t)n) == 0) {
            r = &st->vars[i].value;
        }
        i++;
    }
    return r;
}

static void var_set(struct state *st, const char *name, int n,
                    struct jinja_value v) {
    struct jinja_value *slot = var_find(st, name, n);
    if (slot) {
        *slot = v;
    } else {
        if (st->n_vars == st->cap_vars) {
            int nc = st->cap_vars ? st->cap_vars * 2 : 16;
            st->vars = (struct jinja_var *)xresize(st->vars,
                              (size_t)nc * sizeof(struct jinja_var));
            st->cap_vars = nc;
        }
        st->vars[st->n_vars].name = arena_dup(st, name, (size_t)n);
        st->vars[st->n_vars].value = v;
        st->n_vars++;
    }
}

static void skip_ws(struct state *st) {
    while (*st->p == ' ' || *st->p == '\t' || *st->p == '\n' ||
           *st->p == '\r') {
        st->p++;
    }
}

static const char *unescape(struct state *st, const char *s, size_t n,
                            size_t *outn) {
    char *r = (char *)arena_alloc(st, n + 1);
    size_t w = 0;
    size_t i = 0;
    while (i < n) {
        char c = s[i];
        if (c == '\\' && i + 1 < n) {
            i++;
            char e = s[i];
            c = e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e;
        }
        r[w++] = c;
        i++;
    }
    r[w] = '\0';
    *outn = w;
    return r;
}

/* One expression token into st->{kind,ts,tn,iv}; }} and %} (with optional
 * - prefix) become TK_END without advancing past the marker. */
static void lex(struct state *st) {
    skip_ws(st);
    const char *p = st->p;
    char c = *p;
    st->ts = p;
    st->tn = 0;
    if ((c == '-' && (p[1] == '}' || (p[1] == '%' && p[2] == '}'))) ||
        (c == '}' && !st->brace) || (c == '%' && p[1] == '}')) {
        st->kind = TK_END;
    } else if (c == '\'' || c == '"') {
        const char *e = p + 1;
        while (*e && *e != c) {
            e += (*e == '\\' && e[1]) ? 2 : 1;
        }
        size_t un = 0;
        st->ts = unescape(st, p + 1, (size_t)(e - (p + 1)), &un);
        st->tn = (int)un;
        st->kind = TK_STR;
        st->p = *e ? e + 1 : e;
    } else if (c >= '0' && c <= '9') {
        char *e = NULL;
        st->iv = strtol(p, &e, 10);
        st->kind = TK_INT;
        st->p = e;
    } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               c == '_') {
        const char *e = p;
        while ((*e >= 'a' && *e <= 'z') || (*e >= 'A' && *e <= 'Z') ||
               (*e >= '0' && *e <= '9') || *e == '_') {
            e++;
        }
        st->ts = p;
        st->tn = (int)(e - p);
        st->kind = TK_ID;
        st->p = e;
    } else {
        int two = (p[1] == '=' && (c == '=' || c == '!' || c == '<' ||
                                   c == '>')) ||
                  (c == '*' && p[1] == '*') || (c == '/' && p[1] == '/');
        st->tn = two ? 2 : 1;
        st->kind = c ? TK_OP : TK_END;
        st->p = p + st->tn;
    }
}

static int is_id(struct state *st, const char *w) {
    return st->kind == TK_ID && (int)strlen(w) == st->tn &&
           memcmp(st->ts, w, (size_t)st->tn) == 0;
}

static int is_op(struct state *st, const char *w) {
    return st->kind == TK_OP && (int)strlen(w) == st->tn &&
           memcmp(st->ts, w, (size_t)st->tn) == 0;
}

static struct jinja_value dict_make(struct state *st, int n) {
    struct dict *d = (struct dict *)arena_alloc(st, sizeof(struct dict));
    d->e = (struct jinja_var *)arena_alloc(st,
                   (size_t)n * sizeof(struct jinja_var));
    d->n = 0;
    d->cap = n;
    struct jinja_value v = jv_none();
    v.kind = JV_NODE;
    v.tag = DICT_TAG;
    v.node = d;
    return v;
}

static void dict_put(struct state *st, struct jinja_value dv, const char *k,
                     struct jinja_value val) {
    struct dict *d = (struct dict *)dv.node;
    int i = 0;
    int hit = 0;
    while (i < d->n && !hit) {
        if (strcmp(d->e[i].name, k) == 0) {
            d->e[i].value = val;
            hit = 1;
        }
        i++;
    }
    if (!hit && d->n == d->cap) {
        int ncap = d->cap ? d->cap * 2 : 4;
        struct jinja_var *ne = (struct jinja_var *)arena_alloc(st,
                       (size_t)ncap * sizeof(struct jinja_var));
        for (int j = 0; j < d->n; j++) { ne[j] = d->e[j]; }
        d->e = ne;
        d->cap = ncap;
    }
    if (!hit) {
        d->e[d->n].name = k;
        d->e[d->n].value = val;
        d->n++;
    }
}

static struct jinja_value dict_get(struct jinja_value dv, const char *k,
                                   int n) {
    struct dict *d = (struct dict *)dv.node;
    struct jinja_value r = jv_undef();
    int i = 0;
    while (i < d->n) {
        if ((int)strlen(d->e[i].name) == n &&
            memcmp(d->e[i].name, k, (size_t)n) == 0) {
            r = d->e[i].value;
        }
        i++;
    }
    return r;
}

static int is_dict(struct jinja_value v) {
    return v.kind == JV_NODE && v.tag == DICT_TAG;
}

static struct jinja_value get_attr(struct state *st, struct jinja_value obj,
                                   const char *name, int n) {
    struct jinja_value r = jv_undef();
    if (is_dict(obj)) {
        r = dict_get(obj, name, n);
    } else if (obj.kind == JV_NODE) {
        r = st->host->get(st->host->ctx, obj,
                          arena_dup(st, name, (size_t)n));
    }
    return r;
}

static long seq_len(struct state *st, struct jinja_value v) {
    long n = 0;
    if (v.kind == JV_LIST) {
        n = ((const struct list *)v.node)->n;
    } else if (v.kind == JV_STR) {
        n = (long)strlen(v.s);
    } else if (is_dict(v)) {
        n = ((const struct dict *)v.node)->n;
    } else if (v.kind == JV_NODE) {
        n = st->host->len(st->host->ctx, v);
    }
    return n;
}

static struct jinja_value seq_index(struct state *st, struct jinja_value v,
                                    long i) {
    struct jinja_value r = jv_undef();
    long n = seq_len(st, v);
    if (i < 0) {
        i += n;
    }
    if (i >= 0 && i < n) {
        if (is_dict(v)) {
            r = jv_str(((const struct dict *)v.node)->e[i].name);
        } else if (v.kind == JV_LIST) {
            r = ((const struct list *)v.node)->items[i];
        } else if (v.kind == JV_NODE) {
            r = st->host->index(st->host->ctx, v, i);
        } else if (v.kind == JV_STR) {
            r = jv_str(arena_dup(st, v.s + i, 1));
        }
    }
    return r;
}

static int cmp_jv(struct state *st, struct jinja_value a,
                  struct jinja_value b);

/* The function-style builtins call_named resolves by name. A bare
 * reference to one is "defined" -- Jinja registers them as globals -- even
 * though the engine stores no variable for them; the resolver tags a
 * missed lookup of such a name onto the undefined value's `s` so the
 * defined/undefined tests can see it. */
static int is_builtin_global(const char *nm, int n) {
    int r = (n == 12 && !memcmp(nm, "strftime_now", 12)) ||
            (n == 5 && !memcmp(nm, "range", 5)) ||
            (n == 9 && !memcmp(nm, "namespace", 9)) ||
            (n == 15 && !memcmp(nm, "raise_exception", 15));
    return r;
}

/* `arg` is the test's operand for the comparison family (equalto, ne,
 * lessthan, ...), with has_arg set; argless tests ignore it. The
 * comparison tests arrive via selectattr/rejectattr's 2nd/3rd positional
 * args. The bare `x is equalto y` grammar isn't parsed, so a comparison
 * test reaching here without an argument is an unsupported usage. */
static int do_test(struct state *st, struct jinja_value v,
                   const char *name, int n, struct jinja_value arg,
                   int has_arg) {
    int r = 0;
    int glob = v.kind == JV_UNDEFINED && v.s &&
               is_builtin_global(v.s, (int)strlen(v.s));
    if (n == 6 && !memcmp(name, "string", 6)) {
        r = v.kind == JV_STR;
    } else if (n == 4 && !memcmp(name, "none", 4)) {
        r = v.kind == JV_NONE;
    } else if (n == 9 && !memcmp(name, "undefined", 9)) {
        r = v.kind == JV_UNDEFINED && !glob;
    } else if (n == 7 && !memcmp(name, "defined", 7)) {
        r = v.kind != JV_UNDEFINED || glob;
    } else if (n == 4 && !memcmp(name, "true", 4)) {
        r = v.kind == JV_BOOL && v.i;
    } else if (n == 5 && !memcmp(name, "false", 5)) {
        r = v.kind == JV_BOOL && !v.i;
    } else if (n == 7 && !memcmp(name, "boolean", 7)) {
        r = v.kind == JV_BOOL;
    } else if ((n == 6 && !memcmp(name, "number", 6)) ||
               (n == 7 && !memcmp(name, "integer", 7))) {
        r = v.kind == JV_INT;
    } else if (n == 7 && !memcmp(name, "mapping", 7)) {
        r = v.kind == JV_NODE &&
            st->host->test(st->host->ctx, v, "mapping");
    } else if ((n == 8 && !memcmp(name, "iterable", 8)) ||
               (n == 8 && !memcmp(name, "sequence", 8))) {
        r = v.kind == JV_LIST || v.kind == JV_STR ||
            (v.kind == JV_NODE &&
             st->host->test(st->host->ctx, v, "sequence"));
    } else if (n == 4 && !memcmp(name, "even", 4)) {
        r = v.kind == JV_INT && (v.i % 2) == 0;
    } else if (n == 3 && !memcmp(name, "odd", 3)) {
        r = v.kind == JV_INT && (v.i % 2) != 0;
    } else if (!has_arg && ((n == 7 && !memcmp(name, "equalto", 7)) ||
               (n == 2 && (!memcmp(name, "eq", 2) ||
                !memcmp(name, "ne", 2) || !memcmp(name, "lt", 2) ||
                !memcmp(name, "gt", 2) || !memcmp(name, "le", 2) ||
                !memcmp(name, "ge", 2))) ||
               (n == 8 && !memcmp(name, "lessthan", 8)) ||
               (n == 11 && !memcmp(name, "greaterthan", 11)) ||
               (n == 17 && !memcmp(name, "lessthanorequalto", 17)) ||
               (n == 20 && !memcmp(name, "greaterthanorequalto", 20)))) {
        unsupported(st, "test", name, n);
    } else if ((n == 7 && !memcmp(name, "equalto", 7)) ||
               (n == 2 && !memcmp(name, "eq", 2))) {
        r = jv_eq(v, arg);
    } else if (n == 2 && !memcmp(name, "ne", 2)) {
        r = !jv_eq(v, arg);
    } else if ((n == 8 && !memcmp(name, "lessthan", 8)) ||
               (n == 2 && !memcmp(name, "lt", 2))) {
        r = cmp_jv(st, v, arg) < 0;
    } else if ((n == 11 && !memcmp(name, "greaterthan", 11)) ||
               (n == 2 && !memcmp(name, "gt", 2))) {
        r = cmp_jv(st, v, arg) > 0;
    } else if ((n == 17 && !memcmp(name, "lessthanorequalto", 17)) ||
               (n == 2 && !memcmp(name, "le", 2))) {
        r = cmp_jv(st, v, arg) <= 0;
    } else if ((n == 20 && !memcmp(name, "greaterthanorequalto", 20)) ||
               (n == 2 && !memcmp(name, "ge", 2))) {
        r = cmp_jv(st, v, arg) >= 0;
    } else if (v.kind == JV_NODE) {
        r = st->host->test(st->host->ctx, v,
                           arena_dup(st, name, (size_t)n));
    } else {
        unsupported(st, "test", name, n);
    }
    return r;
}

static void json_str(struct state *st, const char *s) {
    out_putn(st, "\"", 1);
    while (*s) {
        char c = *s;
        const char *esc = c == '"' ? "\\\"" : c == '\\' ? "\\\\"
                        : c == '\n' ? "\\n" : c == '\r' ? "\\r"
                        : c == '\t' ? "\\t" : NULL;
        if (esc) {
            out_putn(st, esc, 2);
        } else {
            out_putn(st, s, 1);
        }
        s++;
    }
    out_putn(st, "\"", 1);
}

static void json_val(struct state *st, struct jinja_value v);

static void json_val(struct state *st, struct jinja_value v) {
    if (v.kind == JV_STR) {
        json_str(st, v.s);
    } else if (v.kind == JV_INT) {
        out_putn(st, to_str(st, v), strlen(to_str(st, v)));
    } else if (v.kind == JV_BOOL) {
        out_putn(st, v.i ? "true" : "false", v.i ? 4 : 5);
    } else if (v.kind == JV_NONE || v.kind == JV_UNDEFINED) {
        out_putn(st, "null", 4);
    } else if (v.kind == JV_LIST) {
        const struct list *l = (const struct list *)v.node;
        int i = 0;
        out_putn(st, "[", 1);
        while (i < l->n) {
            if (i) { out_putn(st, ", ", 2); }
            json_val(st, l->items[i]);
            i++;
        }
        out_putn(st, "]", 1);
    }
}

/* tojson serialises into a scratch buffer (a captured sub-render), then
 * returns it as a string; NODE serialisation is delegated to the host. */
static struct jinja_value to_json(struct state *st, struct jinja_value v) {
    struct jinja_value r = jv_str("");
    if (v.kind == JV_NODE && !is_dict(v)) {
        r = st->host->method(st->host->ctx, v, "tojson", jv_none());
    } else {
        size_t mark = st->out.len;
        json_val(st, v);
        r = jv_str(st->out.len > mark
                   ? arena_dup(st, st->out.buf + mark, st->out.len - mark)
                   : "");
        st->out.len = mark;
        if (st->out.buf) { st->out.buf[mark] = '\0'; }
    }
    return r;
}

static const char *strip(struct state *st, const char *s, const char *set,
                         int left, int right) {
    const char *a = s;
    const char *b = s + strlen(s);
    if (left) {
        while (*a && strchr(set, *a)) {
            a++;
        }
    }
    if (right) {
        while (b > a && strchr(set, b[-1])) {
            b--;
        }
    }
    return arena_dup(st, a, (size_t)(b - a));
}

static const char *replace_str(struct state *st, const char *s,
                               const char *from, const char *to) {
    size_t fl = strlen(from);
    const char *r = arena_dup(st, s, strlen(s));
    if (fl) {
        size_t cnt = 0;
        const char *p = s;
        const char *hit;
        while ((hit = strstr(p, from))) { cnt++; p = hit + fl; }
        size_t tl = strlen(to);
        size_t sl = strlen(s);
        char *out = (char *)arena_alloc(st, (sl - cnt * fl) + cnt * tl + 1);
        char *w = out;
        p = s;
        while ((hit = strstr(p, from))) {
            memcpy(w, p, (size_t)(hit - p));
            w += hit - p;
            memcpy(w, to, tl);
            w += tl;
            p = hit + fl;
        }
        strcpy(w, p);
        r = out;
    }
    return r;
}

static struct jinja_value split_str(struct state *st, const char *s,
                                    const char *sep) {
    struct list *l = list_make(st, 8);
    size_t sl = strlen(sep);
    const char *p = s;
    int cap = 8;
    while (sl) {
        const char *hit = strstr(p, sep);
        const char *end = hit ? hit : p + strlen(p);
        if (l->n == cap) {
            cap *= 2;
            struct jinja_value *ni = (struct jinja_value *)arena_alloc(st,
                              (size_t)cap * sizeof(struct jinja_value));
            memcpy(ni, l->items, (size_t)l->n * sizeof(*ni));
            l->items = ni;
        }
        l->items[l->n++] = jv_str(arena_dup(st, p, (size_t)(end - p)));
        p = hit ? hit + sl : end;
        sl = hit ? sl : 0;
    }
    return jv_listv(l);
}

static struct jinja_value eval_expr(struct state *st);

/* Reverse slice [::-1] materialises a list so the host's element count
 * (also kept in value.i) is never clobbered by an engine flag. */
static struct jinja_value reverse_seq(struct state *st,
                                      struct jinja_value v) {
    long n = seq_len(st, v);
    struct list *l = list_make(st, (int)(n ? n : 1));
    long k = 0;
    while (k < n) {
        l->items[k] = seq_index(st, v, n - 1 - k);
        k++;
    }
    l->n = (int)n;
    return jv_listv(l);
}

static struct jinja_value make_pair(struct state *st,
                                    struct jinja_value k,
                                    struct jinja_value v) {
    struct list *l = list_make(st, 2);
    l->items[0] = k;
    l->items[1] = v;
    l->n = 2;
    return jv_listv(l);
}

/* Mapping -> list of [key, value] pairs (engine dict or host mapping),
 * for dictsort / items / two-variable for. `sorted` orders by key. */
static struct jinja_value to_pairs(struct state *st, struct jinja_value v,
                                   int sorted) {
    long m = is_dict(v) ? ((const struct dict *)v.node)->n
           : v.kind == JV_NODE ? st->host->len(st->host->ctx, v) : 0;
    struct list *l = list_make(st, (int)(m ? m : 1));
    long i = 0;
    while (i < m) {
        struct jinja_value key;
        struct jinja_value val;
        if (is_dict(v)) {
            const struct dict *d = (const struct dict *)v.node;
            key = jv_str(d->e[i].name);
            val = d->e[i].value;
        } else {
            key = st->host->index(st->host->ctx, v, i);
            val = get_attr(st, v, as_str(key), (int)strlen(as_str(key)));
        }
        l->items[i] = make_pair(st, key, val);
        i++;
    }
    l->n = (int)m;
    i = 1;
    while (sorted && i < m) {
        struct jinja_value pair = l->items[i];
        const char *key = as_str(((struct list *)pair.node)->items[0]);
        long j = i - 1;
        while (j >= 0 && strcmp(as_str(((struct list *)
               l->items[j].node)->items[0]), key) > 0) {
            l->items[j + 1] = l->items[j];
            j--;
        }
        l->items[j + 1] = pair;
        i++;
    }
    return jv_listv(l);
}

/* Force any sequence into an engine list (list filter); a mapping yields
 * its keys, matching Jinja's list(dict). */
static struct jinja_value materialize(struct state *st,
                                      struct jinja_value v) {
    struct jinja_value r = v;
    if (v.kind != JV_LIST) {
        long n = seq_len(st, v);
        struct list *l = list_make(st, (int)(n ? n : 1));
        long i = 0;
        while (i < n) {
            l->items[i] = seq_index(st, v, i);
            i++;
        }
        l->n = (int)n;
        r = jv_listv(l);
    }
    return r;
}

static struct jinja_value call_method(struct state *st,
                                      struct jinja_value obj,
                                      const char *nm, int n,
                                      struct jinja_value *a, int na) {
    struct jinja_value r = jv_undef();
    const char *s = as_str(obj);
    const char *a0 = na > 0 ? as_str(a[0]) : "";
    if (n == 10 && !memcmp(nm, "startswith", 10)) {
        r = jv_bool(strncmp(s, a0, strlen(a0)) == 0);
    } else if (n == 8 && !memcmp(nm, "endswith", 8)) {
        size_t ls = strlen(s);
        size_t la = strlen(a0);
        r = jv_bool(ls >= la && !memcmp(s + ls - la, a0, la));
    } else if (n == 5 && !memcmp(nm, "split", 5)) {
        r = split_str(st, s, na > 0 ? a0 : " ");
    } else if (n == 6 && !memcmp(nm, "lstrip", 6)) {
        r = jv_str(strip(st, s, na > 0 ? a0 : " \t\n\r", 1, 0));
    } else if (n == 6 && !memcmp(nm, "rstrip", 6)) {
        r = jv_str(strip(st, s, na > 0 ? a0 : " \t\n\r", 0, 1));
    } else if (n == 5 && !memcmp(nm, "strip", 5)) {
        r = jv_str(strip(st, s, na > 0 ? a0 : " \t\n\r", 1, 1));
    } else if (n == 7 && !memcmp(nm, "replace", 7)) {
        r = na > 1 ? jv_str(replace_str(st, s, a0, as_str(a[1]))) : obj;
    } else if (n == 5 && !memcmp(nm, "items", 5)) {
        r = to_pairs(st, obj, 0);
    } else if (n == 3 && !memcmp(nm, "get", 3)) {
        struct jinja_value g = get_attr(st, obj, a0, (int)strlen(a0));
        r = g.kind == JV_UNDEFINED ? (na > 1 ? a[1] : jv_none()) : g;
    } else if (n == 4 && !memcmp(nm, "keys", 4)) {
        r = materialize(st, obj);
    } else if (obj.kind == JV_NODE) {
        r = st->host->method(st->host->ctx, obj,
                             arena_dup(st, nm, (size_t)n),
                             na > 0 ? a[0] : jv_none());
    } else {
        unsupported(st, "method", nm, n);
    }
    return r;
}

/* Python-semantics slice for [start:stop:step]; missing bounds default by
 * step direction. Materialises the selected elements into a new list. */
static struct jinja_value slice_seq(struct state *st, struct jinja_value v,
                                    int hstart, long start, int hstop,
                                    long stop, int hstep, long step) {
    long len = seq_len(st, v);
    long lo;
    long hi;
    long i;
    struct list *l;
    if (!hstep) { step = 1; }
    if (step == 0) { fail(st, "slice step cannot be zero"); }
    lo = step < 0 ? -1 : 0;
    hi = step < 0 ? len - 1 : len;
    start = hstart ? (start < 0 ? start + len : start)
                   : (step < 0 ? len - 1 : 0);
    stop = hstop ? (stop < 0 ? stop + len : stop)
                 : (step < 0 ? -1 : len);
    if (start < lo) { start = lo; }
    if (start > hi) { start = hi; }
    if (stop < lo) { stop = lo; }
    if (stop > hi) { stop = hi; }
    l = list_make(st, (int)(len ? len : 1));
    i = start;
    while (step < 0 ? i > stop : i < stop) {
        if (i >= 0 && i < len) { l->items[l->n++] = seq_index(st, v, i); }
        i += step;
    }
    return jv_listv(l);
}

static struct jinja_value apply_filter(struct state *st,
                                       struct jinja_value v, const char *nm,
                                       int n, struct callargs *ca);

static int cmp_jv(struct state *st, struct jinja_value a,
                  struct jinja_value b) {
    int r;
    if (a.kind == JV_INT && b.kind == JV_INT) {
        r = a.i < b.i ? -1 : a.i > b.i;
    } else {
        r = strcmp(to_str(st, a), to_str(st, b));
    }
    return r;
}

static struct jinja_value kw_get(struct callargs *ca, const char *name) {
    struct jinja_value r = jv_undef();
    int n = (int)strlen(name);
    int i = 0;
    while (i < ca->nkw) {
        if (ca->kw[i].nlen == n &&
            !memcmp(ca->kw[i].name, name, (size_t)n)) {
            r = ca->kw[i].val;
        }
        i++;
    }
    return r;
}

static struct jinja_value sort_key(struct state *st, struct jinja_value it,
                                   struct jinja_value attr) {
    return attr.kind == JV_STR
        ? get_attr(st, it, attr.s, (int)strlen(attr.s)) : it;
}

static struct jinja_value filter_sort(struct state *st, struct jinja_value v,
                                      struct jinja_value attr, int rev) {
    struct jinja_value lv = materialize(st, v);
    struct list *l = (struct list *)lv.node;
    int i = 1;
    while (i < l->n) {
        struct jinja_value cur = l->items[i];
        struct jinja_value ck = sort_key(st, cur, attr);
        int j = i - 1;
        while (j >= 0 && (rev
               ? cmp_jv(st, sort_key(st, l->items[j], attr), ck) < 0
               : cmp_jv(st, sort_key(st, l->items[j], attr), ck) > 0)) {
            l->items[j + 1] = l->items[j];
            j--;
        }
        l->items[j + 1] = cur;
        i++;
    }
    return lv;
}

/* select/reject/selectattr/rejectattr. When `byattr`, pos[0] names the
 * attribute, an optional pos[1] names a test on it, and pos[2] is the
 * test's argument; otherwise pos[0] names the test and pos[1] its
 * argument. No test = keep truthy. `rej` inverts the predicate. */
static struct jinja_value filter_select(struct state *st,
                                        struct jinja_value v,
                                        struct callargs *ca, int byattr,
                                        int rej) {
    struct jinja_value attr = byattr && ca->npos > 0 ? ca->pos[0]
                                                     : jv_undef();
    int ti = byattr ? 1 : 0;
    struct jinja_value tst = ca->pos[ti];
    int has_tst = ca->npos > ti;
    struct jinja_value targ = ca->npos > ti + 1 ? ca->pos[ti + 1]
                                                : jv_undef();
    int has_arg = ca->npos > ti + 1;
    long ln = seq_len(st, v);
    struct list *l = list_make(st, (int)(ln ? ln : 1));
    long i = 0;
    while (i < ln) {
        struct jinja_value e = seq_index(st, v, i);
        struct jinja_value t = sort_key(st, e, attr);
        int keep = has_tst ? do_test(st, t, tst.s, (int)strlen(tst.s),
                                     targ, has_arg) : truthy(st, t);
        if (keep ^ rej) { l->items[l->n++] = e; }
        i++;
    }
    return jv_listv(l);
}

static struct jinja_value filter_groupby(struct state *st,
                                         struct jinja_value v,
                                         struct jinja_value attr) {
    struct jinja_value sv = filter_sort(st, v, attr, 0);
    struct list *s = (struct list *)sv.node;
    struct list *out = list_make(st, (int)(s->n ? s->n : 1));
    int i = 0;
    while (i < s->n) {
        struct jinja_value k = sort_key(st, s->items[i], attr);
        struct list *grp = list_make(st, s->n - i);
        while (i < s->n &&
               cmp_jv(st, sort_key(st, s->items[i], attr), k) == 0) {
            grp->items[grp->n++] = s->items[i];
            i++;
        }
        out->items[out->n++] = make_pair(st, k, jv_listv(grp));
    }
    return jv_listv(out);
}

static struct jinja_value filter_minmax(struct state *st,
                                        struct jinja_value v, int want_max) {
    long ln = seq_len(st, v);
    struct jinja_value r = ln > 0 ? seq_index(st, v, 0) : jv_undef();
    long i = 1;
    while (i < ln) {
        struct jinja_value e = seq_index(st, v, i);
        int c = cmp_jv(st, e, r);
        if (want_max ? c > 0 : c < 0) { r = e; }
        i++;
    }
    return r;
}

static long filter_sum(struct state *st, struct jinja_value v) {
    long ln = seq_len(st, v);
    long s = 0;
    long i = 0;
    while (i < ln) {
        struct jinja_value e = seq_index(st, v, i);
        s += e.kind == JV_INT ? e.i : 0;
        i++;
    }
    return s;
}

static struct jinja_value filter_unique(struct state *st,
                                        struct jinja_value v) {
    long ln = seq_len(st, v);
    struct list *l = list_make(st, (int)(ln ? ln : 1));
    long i = 0;
    while (i < ln) {
        struct jinja_value e = seq_index(st, v, i);
        int seen = 0;
        int j = 0;
        while (j < l->n && !seen) {
            seen = jv_eq(l->items[j], e);
            j++;
        }
        if (!seen) { l->items[l->n++] = e; }
        i++;
    }
    return jv_listv(l);
}

static const char *str_title(struct state *st, const char *s) {
    char *t = (char *)arena_dup(st, s, strlen(s));
    char *q = t;
    int start = 1;
    while (*q) {
        *q = (char)(start ? toupper((unsigned char)*q)
                          : tolower((unsigned char)*q));
        start = !isalnum((unsigned char)*q);
        q++;
    }
    return t;
}

static struct jinja_value apply_filter(struct state *st,
                                       struct jinja_value v, const char *nm,
                                       int n, struct callargs *ca) {
    struct jinja_value r = v;
    struct jinja_value a0 = ca->npos > 0 ? ca->pos[0] : jv_undef();
    if (n == 4 && !memcmp(nm, "trim", 4)) {
        r = jv_str(strip(st, to_str(st, v), " \t\n\r", 1, 1));
    } else if (n == 6 && !memcmp(nm, "length", 6)) {
        r = jv_int(seq_len(st, v));
    } else if (n == 6 && !memcmp(nm, "tojson", 6)) {
        r = to_json(st, v);
    } else if (n == 4 && !memcmp(nm, "safe", 4)) {
        r = v;
    } else if (n == 6 && !memcmp(nm, "string", 6)) {
        r = jv_str(to_str(st, v));
    } else if (n == 5 && !memcmp(nm, "upper", 5)) {
        char *t = (char *)arena_dup(st, to_str(st, v), strlen(to_str(st,
                  v)));
        char *q = t;
        while (*q) { *q = (char)toupper((unsigned char)*q); q++; }
        r = jv_str(t);
    } else if (n == 5 && !memcmp(nm, "lower", 5)) {
        char *t = (char *)arena_dup(st, to_str(st, v), strlen(to_str(st,
                  v)));
        char *q = t;
        while (*q) { *q = (char)tolower((unsigned char)*q); q++; }
        r = jv_str(t);
    } else if (n == 7 && !memcmp(nm, "default", 7)) {
        int falsy = ca->npos > 1 && truthy(st, ca->pos[1]);
        int rep = falsy ? !truthy(st, v) : v.kind == JV_UNDEFINED;
        r = (rep && ca->npos > 0) ? a0 : v;
    } else if (n == 5 && !memcmp(nm, "first", 5)) {
        r = seq_index(st, v, 0);
    } else if (n == 4 && !memcmp(nm, "last", 4)) {
        r = seq_index(st, v, seq_len(st, v) - 1);
    } else if (n == 8 && !memcmp(nm, "dictsort", 8)) {
        r = to_pairs(st, v, 1);
    } else if (n == 5 && !memcmp(nm, "items", 5)) {
        r = to_pairs(st, v, 0);
    } else if (n == 4 && !memcmp(nm, "list", 4)) {
        r = materialize(st, v);
    } else if (n == 4 && !memcmp(nm, "join", 4)) {
        const char *sep = ca->npos > 0 ? as_str(a0) : "";
        long ln = seq_len(st, v);
        long i = 0;
        size_t mark = st->out.len;
        while (i < ln) {
            if (i) { out_putn(st, sep, strlen(sep)); }
            const char *e = to_str(st, seq_index(st, v, i));
            out_putn(st, e, strlen(e));
            i++;
        }
        r = jv_str(st->out.len > mark
                   ? arena_dup(st, st->out.buf + mark, st->out.len - mark)
                   : "");
        st->out.len = mark;
        if (st->out.buf) { st->out.buf[mark] = '\0'; }
    } else if (n == 3 && !memcmp(nm, "map", 3)) {
        struct jinja_value attr = kw_get(ca, "attribute");
        const char *fn = ca->npos > 0 ? as_str(a0) : "";
        long ln = seq_len(st, v);
        struct list *l = list_make(st, (int)(ln ? ln : 1));
        struct callargs sub = {0};
        long i = 0;
        while (i < ln) {
            struct jinja_value e = seq_index(st, v, i);
            l->items[i] = attr.kind == JV_STR
                ? get_attr(st, e, attr.s, (int)strlen(attr.s))
                : apply_filter(st, e, fn, (int)strlen(fn), &sub);
            i++;
        }
        l->n = (int)ln;
        r = jv_listv(l);
    } else if (n == 7 && !memcmp(nm, "replace", 7)) {
        r = ca->npos > 1 ? jv_str(replace_str(st, to_str(st, v),
                           as_str(a0), as_str(ca->pos[1]))) : v;
    } else if (n == 10 && !memcmp(nm, "capitalize", 10)) {
        char *t = (char *)arena_dup(st, to_str(st, v),
                                    strlen(to_str(st, v)));
        if (t[0]) { t[0] = (char)toupper((unsigned char)t[0]); }
        r = jv_str(t);
    } else if (n == 3 && !memcmp(nm, "int", 3)) {
        r = jv_int(v.kind == JV_INT ? v.i : strtol(to_str(st, v), NULL,
                   10));
    } else if (n == 5 && !memcmp(nm, "count", 5)) {
        r = jv_int(seq_len(st, v));
    } else if (n == 7 && !memcmp(nm, "reverse", 7)) {
        r = reverse_seq(st, v);
    } else if (n == 4 && !memcmp(nm, "sort", 4)) {
        r = filter_sort(st, v, kw_get(ca, "attribute"),
                        truthy(st, kw_get(ca, "reverse")));
    } else if (n == 6 && !memcmp(nm, "unique", 6)) {
        r = filter_unique(st, v);
    } else if (n == 3 && !memcmp(nm, "min", 3)) {
        r = filter_minmax(st, v, 0);
    } else if (n == 3 && !memcmp(nm, "max", 3)) {
        r = filter_minmax(st, v, 1);
    } else if (n == 3 && !memcmp(nm, "sum", 3)) {
        r = jv_int(filter_sum(st, v));
    } else if (n == 3 && !memcmp(nm, "abs", 3)) {
        r = jv_int(v.i < 0 ? -v.i : v.i);
    } else if (n == 5 && !memcmp(nm, "title", 5)) {
        r = jv_str(str_title(st, to_str(st, v)));
    } else if (n == 6 && !memcmp(nm, "select", 6)) {
        r = filter_select(st, v, ca, 0, 0);
    } else if (n == 6 && !memcmp(nm, "reject", 6)) {
        r = filter_select(st, v, ca, 0, 1);
    } else if (n == 10 && !memcmp(nm, "selectattr", 10)) {
        r = filter_select(st, v, ca, 1, 0);
    } else if (n == 10 && !memcmp(nm, "rejectattr", 10)) {
        r = filter_select(st, v, ca, 1, 1);
    } else if (n == 7 && !memcmp(nm, "groupby", 7)) {
        r = filter_groupby(st, v, a0);
    } else {
        unsupported(st, "filter", nm, n);
    }
    return r;
}

static struct macro *find_macro(struct state *st, const char *nm, int n) {
    struct macro *r = NULL;
    int i = 0;
    while (i < st->n_macros && !r) {
        if (st->macros[i].nlen == n &&
            !memcmp(st->macros[i].name, nm, (size_t)n)) {
            r = &st->macros[i];
        }
        i++;
    }
    return r;
}

static int render_block(struct state *st);
static void consume_close(struct state *st);

static int parse_args(struct state *st, struct jinja_value *out) {
    int na = 0;
    lex(st);
    while (!is_op(st, ")") && st->kind != TK_END && na < MAX_ARGS) {
        out[na++] = eval_expr(st);
        if (is_op(st, ",")) { lex(st); }
    }
    if (is_op(st, ")")) { lex(st); }
    return na;
}

/* Parses a call argument list (current token is the opening paren) into
 * positional values and `name=value` keyword pairs. Keyword detection
 * peeks one token past an identifier and rewinds the lexer if it is not
 * followed by '='. */
static void parse_call(struct state *st, struct callargs *ca) {
    ca->npos = 0;
    ca->nkw = 0;
    lex(st);
    while (!is_op(st, ")") && st->kind != TK_END &&
           ca->npos < MAX_ARGS && ca->nkw < MAX_ARGS) {
        int kw = 0;
        if (st->kind == TK_ID) {
            int k0 = st->kind;
            const char *ts0 = st->ts;
            int tn0 = st->tn;
            long iv0 = st->iv;
            const char *p0 = st->p;
            const char *nm = st->ts;
            int nn = st->tn;
            lex(st);
            if (is_op(st, "=")) {
                lex(st);
                ca->kw[ca->nkw].name = arena_dup(st, nm, (size_t)nn);
                ca->kw[ca->nkw].nlen = nn;
                ca->kw[ca->nkw].val = eval_expr(st);
                ca->nkw++;
                kw = 1;
            } else {
                st->kind = k0;
                st->ts = ts0;
                st->tn = tn0;
                st->iv = iv0;
                st->p = p0;
            }
        }
        if (!kw) {
            ca->pos[ca->npos++] = eval_expr(st);
        }
        if (is_op(st, ",")) { lex(st); }
    }
    if (is_op(st, ")")) { lex(st); }
}

static struct jinja_value call_macro(struct state *st, struct macro *m,
                                     struct callargs *ca) {
    struct jinja_value saved[MAX_PARAMS];
    struct jinja_value loopsave = jv_undef();
    struct jinja_value *lp;
    /* A call inside a dead (muted) branch has no observable effect: its
     * output is suppressed and macros don't mutate the enclosing scope.
     * Rendering it anyway would run the recursive calls in not-taken
     * branches of self-recursive macros, recursing without bound. The
     * arguments were already evaluated by parse_call (harmless data
     * access), so returning here leaves the parser correctly positioned. */
    if (st->mute) {
        return jv_str("");
    }
    lp = var_find(st, "loop", 4);
    int k0 = st->kind;
    const char *ts0 = st->ts;
    int tn0 = st->tn;
    long iv0 = st->iv;
    const char *p0 = st->p;
    int tl0 = st->trim_left;
    int i = 0;
    loopsave = lp ? *lp : jv_undef();
    while (i < m->nparam) {
        struct jinja_value *slot = var_find(st, m->param[i].name,
                                            m->param[i].nlen);
        struct jinja_value val = jv_undef();
        int bound = 0;
        saved[i] = slot ? *slot : jv_undef();
        if (i < ca->npos) {
            val = ca->pos[i];
            bound = 1;
        }
        int j = 0;
        while (!bound && j < ca->nkw) {
            if (ca->kw[j].nlen == m->param[i].nlen &&
                !memcmp(ca->kw[j].name, m->param[i].name,
                        (size_t)m->param[i].nlen)) {
                val = ca->kw[j].val;
                bound = 1;
            }
            j++;
        }
        if (!bound && m->param[i].def) {
            const char *back = st->p;
            st->p = m->param[i].def;
            lex(st);
            val = eval_expr(st);
            st->p = back;
        }
        var_set(st, m->param[i].name, m->param[i].nlen, val);
        i++;
    }
    /* Capture the body's output by rendering into the shared buffer and
     * slicing it back out, rather than swapping in a private malloc'd
     * buffer. A raise inside the body longjmps past any per-call free, so
     * private buffers would leak; the shared buffer is freed once, by
     * jinja_render, on both the success and error paths. */
    size_t mark = st->out.len;
    st->p = m->body;
    st->trim_left = 0;
    render_block(st);
    const char *cap = st->out.len > mark
        ? arena_dup(st, st->out.buf + mark, st->out.len - mark) : "";
    struct jinja_value out = jv_str(cap);
    st->out.len = mark;
    if (st->out.buf) { st->out.buf[mark] = '\0'; }
    st->kind = k0;
    st->ts = ts0;
    st->tn = tn0;
    st->iv = iv0;
    st->p = p0;
    st->trim_left = tl0;
    i = 0;
    while (i < m->nparam) {
        var_set(st, m->param[i].name, m->param[i].nlen, saved[i]);
        i++;
    }
    var_set(st, "loop", 4, loopsave);
    return out;
}

static struct jinja_value call_named(struct state *st, const char *nm,
                                     int n) {
    struct jinja_value r = jv_undef();
    struct callargs ca;
    parse_call(st, &ca);
    struct macro *m = find_macro(st, nm, n);
    if (n == 9 && !memcmp(nm, "namespace", 9)) {
        r = dict_make(st, ca.nkw > 0 ? ca.nkw : 1);
        int i = 0;
        while (i < ca.nkw) {
            dict_put(st, r, ca.kw[i].name, ca.kw[i].val);
            i++;
        }
    } else if (m) {
        r = call_macro(st, m, &ca);
    } else if (n == 15 && !memcmp(nm, "raise_exception", 15)) {
        if (!st->mute) {
            fail(st, ca.npos > 0 ? as_str(ca.pos[0]) : "raise_exception");
        }
    } else if (n == 5 && !memcmp(nm, "range", 5)) {
        long a = ca.npos >= 2 ? ca.pos[0].i : 0;
        long b = ca.npos >= 2 ? ca.pos[1].i
               : ca.npos >= 1 ? ca.pos[0].i : 0;
        long step = ca.npos >= 3 ? ca.pos[2].i : 1;
        long cnt = 0;
        if (step == 0) { fail(st, "range() step cannot be zero"); }
        if (step > 0 && b > a) { cnt = (b - a + step - 1) / step; }
        if (step < 0 && b < a) { cnt = (a - b - step - 1) / -step; }
        struct list *l = list_make(st, (int)(cnt ? cnt : 1));
        long i = a;
        long w = 0;
        while (w < cnt) {
            l->items[w] = jv_int(i);
            i += step;
            w++;
        }
        l->n = (int)cnt;
        r = jv_listv(l);
    } else if (n == 12 && !memcmp(nm, "strftime_now", 12)) {
        const char *fmt = ca.npos > 0 ? as_str(ca.pos[0]) : "%Y-%m-%d";
        time_t now = time(NULL);
        char buf[256];
        size_t k = strftime(buf, sizeof buf, fmt, localtime(&now));
        r = jv_str(arena_dup(st, buf, k));
    } else {
        unsupported(st, "function", nm, n);
    }
    return r;
}

static struct jinja_value eval_primary(struct state *st) {
    struct jinja_value v = jv_undef();
    if (st->kind == TK_STR) {
        v = jv_str(arena_dup(st, st->ts, (size_t)st->tn));
        lex(st);
    } else if (st->kind == TK_INT) {
        v = jv_int(st->iv);
        lex(st);
    } else if (is_id(st, "true") || is_id(st, "True")) {
        v = jv_bool(1);
        lex(st);
    } else if (is_id(st, "false") || is_id(st, "False")) {
        v = jv_bool(0);
        lex(st);
    } else if (is_id(st, "none") || is_id(st, "None")) {
        v = jv_none();
        lex(st);
    } else if (is_op(st, "(")) {
        lex(st);
        v = eval_expr(st);
        if (is_op(st, ")")) { lex(st); }
    } else if (is_op(st, "[")) {
        lex(st);
        struct list *l = list_make(st, MAX_ARGS);
        int cnt = 0;
        while (!is_op(st, "]") && st->kind != TK_END && cnt < MAX_ARGS) {
            l->items[cnt++] = eval_expr(st);
            if (is_op(st, ",")) { lex(st); }
        }
        l->n = cnt;
        if (is_op(st, "]")) { lex(st); }
        v = jv_listv(l);
    } else if (is_op(st, "{")) {
        struct jinja_value d = dict_make(st, MAX_ARGS);
        st->brace++;
        lex(st);
        while (!is_op(st, "}") && st->kind != TK_END) {
            struct jinja_value k = eval_expr(st);
            const char *ks = arena_dup(st, to_str(st, k),
                                       strlen(to_str(st, k)));
            if (is_op(st, ":")) { lex(st); }
            dict_put(st, d, ks, eval_expr(st));
            if (is_op(st, ",")) { lex(st); }
        }
        st->brace--;
        if (is_op(st, "}")) { lex(st); }
        v = d;
    } else if (st->kind == TK_ID) {
        const char *nm = arena_dup(st, st->ts, (size_t)st->tn);
        int n = st->tn;
        lex(st);
        if (is_op(st, "(")) {
            v = call_named(st, nm, n);
        } else {
            struct jinja_value *slot = var_find(st, nm, n);
            v = slot ? *slot : jv_undef();
            if (!slot && is_builtin_global(nm, n)) { v.s = nm; }
        }
    }
    return v;
}

static struct jinja_value eval_postfix(struct state *st) {
    struct jinja_value v = eval_primary(st);
    while (is_op(st, ".") || is_op(st, "[") || is_op(st, "|")) {
        if (is_op(st, ".")) {
            lex(st);
            const char *nm = arena_dup(st, st->ts, (size_t)st->tn);
            int n = st->tn;
            lex(st);
            if (is_op(st, "(")) {
                struct jinja_value a[MAX_ARGS];
                int na = parse_args(st, a);
                v = call_method(st, v, nm, n, a, na);
            } else {
                v = get_attr(st, v, nm, n);
            }
        } else if (is_op(st, "[")) {
            lex(st);
            struct jinja_value idx = jv_undef();
            long start = 0;
            long stop = 0;
            long step = 0;
            int hstart = 0;
            int hstop = 0;
            int hstep = 0;
            int slice = 0;
            if (!is_op(st, ":")) {
                idx = eval_expr(st);
                hstart = 1;
                start = idx.i;
            }
            if (is_op(st, ":")) {
                slice = 1;
                lex(st);
                if (!is_op(st, ":") && !is_op(st, "]")) {
                    hstop = 1;
                    stop = eval_expr(st).i;
                }
                if (is_op(st, ":")) {
                    lex(st);
                    if (!is_op(st, "]")) {
                        hstep = 1;
                        step = eval_expr(st).i;
                    }
                }
            }
            if (slice) {
                v = slice_seq(st, v, hstart, start, hstop, stop, hstep,
                              step);
            } else {
                v = idx.kind == JV_STR ? get_attr(st, v, idx.s,
                    (int)strlen(idx.s)) : seq_index(st, v, idx.i);
            }
            if (is_op(st, "]")) { lex(st); }
        } else {
            lex(st);
            const char *nm = arena_dup(st, st->ts, (size_t)st->tn);
            int n = st->tn;
            lex(st);
            struct callargs ca = {0};
            if (is_op(st, "(")) {
                parse_call(st, &ca);
            }
            v = apply_filter(st, v, nm, n, &ca);
        }
    }
    return v;
}

static long py_floordiv(long a, long b) {
    long q = a / b;
    long r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) { q--; }
    return q;
}

static long py_mod(long a, long b) {
    long m = a % b;
    if (m != 0 && ((m < 0) != (b < 0))) { m += b; }
    return m;
}

static struct jinja_value eval_unary(struct state *st) {
    struct jinja_value v = jv_undef();
    if (is_op(st, "-")) {
        lex(st);
        v = jv_int(-eval_unary(st).i);
    } else if (is_op(st, "+")) {
        lex(st);
        v = eval_unary(st);
    } else {
        v = eval_postfix(st);
    }
    return v;
}

/* Unlike Python, Jinja binds unary minus into the base of `**` (so
 * -2**2 is (-2)**2 == 4) and `**` is left-associative. Integer only:
 * a negative exponent (Jinja yields a float) collapses to 1. */
static struct jinja_value eval_pow(struct state *st) {
    struct jinja_value v = eval_unary(st);
    while (is_op(st, "**")) {
        lex(st);
        long base = v.i;
        long e = eval_unary(st).i;
        long res = 1;
        while (e > 0) {
            res *= base;
            e--;
        }
        v = jv_int(res);
    }
    return v;
}

/* Integer-only multiplicative level. `/` is integer (no float type), a
 * documented divergence from Jinja's true division; `//` and `%` follow
 * Python's floor/sign rules. Division by zero raises (NULL), as in Jinja. */
static struct jinja_value eval_mul(struct state *st) {
    struct jinja_value v = eval_pow(st);
    while (is_op(st, "*") || is_op(st, "/") || is_op(st, "//") ||
           is_op(st, "%")) {
        int mul = is_op(st, "*");
        int mod = is_op(st, "%");
        lex(st);
        long a = v.i;
        long b = eval_pow(st).i;
        if (!mul && b == 0) { fail(st, "division by zero"); }
        v = jv_int(mul ? a * b : mod ? py_mod(a, b) : py_floordiv(a, b));
    }
    return v;
}

static struct jinja_value eval_concat(struct state *st) {
    struct jinja_value v = eval_mul(st);
    while (is_op(st, "~") || is_op(st, "+") || is_op(st, "-")) {
        int sub = is_op(st, "-");
        int add = is_op(st, "+");
        lex(st);
        struct jinja_value r = eval_mul(st);
        if (!sub && (v.kind == JV_STR || r.kind == JV_STR || add == 0)) {
            const char *a = to_str(st, v);
            const char *b = to_str(st, r);
            size_t la = strlen(a);
            size_t lb = strlen(b);
            char *c = (char *)arena_alloc(st, la + lb + 1);
            memcpy(c, a, la);
            memcpy(c + la, b, lb);
            c[la + lb] = '\0';
            v = jv_str(c);
        } else {
            v = jv_int(sub ? v.i - r.i : v.i + r.i);
        }
    }
    return v;
}

static int contains(struct state *st, struct jinja_value c,
                    struct jinja_value item) {
    int r = 0;
    if (c.kind == JV_STR) {
        r = strstr(c.s, to_str(st, item)) != NULL;
    } else {
        long n = seq_len(st, c);
        long i = 0;
        while (i < n && !r) {
            r = jv_eq(seq_index(st, c, i), item);
            i++;
        }
    }
    return r;
}

static struct jinja_value eval_cmp(struct state *st) {
    struct jinja_value v = eval_concat(st);
    int go = 1;
    while (go) {
        if (is_op(st, "==") || is_op(st, "!=") || is_op(st, "<") ||
            is_op(st, ">") || is_op(st, "<=") || is_op(st, ">=")) {
            int ne = is_op(st, "!=");
            int lt = is_op(st, "<");
            int gt = is_op(st, ">");
            int le = is_op(st, "<=");
            int ge = is_op(st, ">=");
            lex(st);
            struct jinja_value r = eval_concat(st);
            v = jv_bool(lt ? v.i < r.i : gt ? v.i > r.i
                      : le ? v.i <= r.i : ge ? v.i >= r.i
                      : (jv_eq(v, r) ^ ne));
        } else if (is_id(st, "in")) {
            lex(st);
            struct jinja_value r = eval_concat(st);
            v = jv_bool(contains(st, r, v));
        } else if (is_id(st, "not")) {
            lex(st);
            lex(st);
            struct jinja_value r = eval_concat(st);
            v = jv_bool(!contains(st, r, v));
        } else if (is_id(st, "is")) {
            lex(st);
            int neg = is_id(st, "not");
            if (neg) { lex(st); }
            const char *nm = arena_dup(st, st->ts, (size_t)st->tn);
            int n = st->tn;
            lex(st);
            v = jv_bool(do_test(st, v, nm, n, jv_undef(), 0) ^ neg);
        } else {
            go = 0;
        }
    }
    return v;
}

static struct jinja_value eval_not(struct state *st) {
    struct jinja_value v = jv_undef();
    if (is_id(st, "not")) {
        lex(st);
        v = jv_bool(!truthy(st, eval_not(st)));
    } else {
        v = eval_cmp(st);
    }
    return v;
}

static struct jinja_value eval_and(struct state *st) {
    struct jinja_value v = eval_not(st);
    while (is_id(st, "and")) {
        lex(st);
        struct jinja_value r = eval_not(st);
        v = truthy(st, v) ? r : v;
    }
    return v;
}

static struct jinja_value eval_or(struct state *st) {
    struct jinja_value v = eval_and(st);
    while (is_id(st, "or")) {
        lex(st);
        struct jinja_value r = eval_and(st);
        v = truthy(st, v) ? v : r;
    }
    return v;
}

/* Ternary "A if COND else B" is the lowest-precedence form. Both arms are
 * parsed so the cursor advances; only the selected one is returned. */
static struct jinja_value eval_expr(struct state *st) {
    struct jinja_value v = eval_or(st);
    if (is_id(st, "if")) {
        lex(st);
        int cond = truthy(st, eval_or(st));
        struct jinja_value alt = jv_undef();
        if (is_id(st, "else")) {
            lex(st);
            alt = eval_expr(st);
        }
        v = cond ? v : alt;
    }
    return v;
}

static void consume_close(struct state *st) {
    int dash = st->ts[0] == '-';
    st->trim_left = dash;
    st->p = st->ts + (dash ? 3 : 2);
}

static void prime(struct state *st, const char *after_open) {
    st->p = after_open;
    lex(st);
}

static void emit_lit(struct state *st, const char *a, const char *b,
                     int lead) {
    if (st->trim_left) {
        while (a < b && (*a == ' ' || *a == '\t' || *a == '\n' ||
                         *a == '\r')) {
            a++;
        }
    }
    if (lead) {
        while (b > a && (b[-1] == ' ' || b[-1] == '\t' ||
                         b[-1] == '\n' || b[-1] == '\r')) {
            b--;
        }
    }
    out_putn(st, a, (size_t)(b - a));
    st->trim_left = 0;
}

static void do_set(struct state *st) {
    const char *name = arena_dup(st, st->ts, (size_t)st->tn);
    int member = 0;
    const char *key = NULL;
    lex(st);
    if (is_op(st, ".")) {
        lex(st);
        key = arena_dup(st, st->ts, (size_t)st->tn);
        member = 1;
        lex(st);
    }
    struct jinja_value val = jv_str("");
    if (is_op(st, "=")) {
        lex(st);
        val = eval_expr(st);
        consume_close(st);
    } else {
        consume_close(st);
        size_t mark = st->out.len;
        render_block(st);
        val = jv_str(st->out.len > mark
                     ? arena_dup(st, st->out.buf + mark,
                                 st->out.len - mark) : "");
        st->out.len = mark;
        if (st->out.buf) { st->out.buf[mark] = '\0'; }
        consume_close(st);
    }
    if (!st->mute && member) {
        struct jinja_value *ns = var_find(st, name, (int)strlen(name));
        if (ns) { dict_put(st, *ns, key, val); }
        else { fail(st, "set member on undeclared namespace"); }
    } else if (!st->mute) {
        var_set(st, name, (int)strlen(name), val);
    }
}

static void do_filter(struct state *st) {
    const char *nm = arena_dup(st, st->ts, (size_t)st->tn);
    int n = st->tn;
    lex(st);
    struct callargs ca = {0};
    if (is_op(st, "(")) {
        parse_call(st, &ca);
    }
    consume_close(st);
    size_t mark = st->out.len;
    render_block(st);
    struct jinja_value body = jv_str(st->out.len > mark
        ? arena_dup(st, st->out.buf + mark, st->out.len - mark) : "");
    st->out.len = mark;
    if (st->out.buf) { st->out.buf[mark] = '\0'; }
    struct jinja_value r = apply_filter(st, body, nm, n, &ca);
    const char *s = to_str(st, r);
    out_putn(st, s, strlen(s));
    consume_close(st);
}

static void do_macro(struct state *st) {
    /* Past the cap, parse into a throwaway slot so the lexer still advances
       over the definition and body; the macro is just not registered. */
    struct macro scratch;
    int have_room = st->n_macros < MAX_MACROS;
    struct macro *m = have_room ? &st->macros[st->n_macros] : &scratch;
    m->name = arena_dup(st, st->ts, (size_t)st->tn);
    m->nlen = st->tn;
    m->nparam = 0;
    lex(st);
    lex(st);
    while (st->kind == TK_ID && m->nparam < MAX_PARAMS) {
        m->param[m->nparam].name = arena_dup(st, st->ts, (size_t)st->tn);
        m->param[m->nparam].nlen = st->tn;
        m->param[m->nparam].def = NULL;
        lex(st);
        if (is_op(st, "=")) {
            lex(st);
            m->param[m->nparam].def = st->ts;
            eval_expr(st);
        }
        m->nparam++;
        if (is_op(st, ",")) { lex(st); }
    }
    if (is_op(st, ")")) { lex(st); }
    consume_close(st);
    m->body = st->p;
    if (have_room) { st->n_macros++; }
    /* Skip the body by raw scan to the matching endmacro: rendering it
     * here (even muted) would run its if/for terminators with no outer
     * consumer, drifting the mute counter. The body is rendered only
     * when the macro is called. */
    const char *p = st->p;
    int depth = 1;
    while (depth > 0 && *p) {
        if (p[0] == '{' && p[1] == '#') {
            p += 2;
            while (*p && !(p[0] == '#' && p[1] == '}')) { p++; }
            p += *p ? 2 : 0;
        } else if (p[0] == '{' && p[1] == '%') {
            const char *q = p + 2 + (p[2] == '-');
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') {
                q++;
            }
            depth += !strncmp(q, "macro", 5);
            depth -= !strncmp(q, "endmacro", 8);
            while (*q && !(q[0] == '%' && q[1] == '}')) {
                char c = *q;
                q += (c == '\'' || c == '"') ? 1 : 0;
                while ((c == '\'' || c == '"') && *q && *q != c) {
                    q += (*q == '\\' && q[1]) ? 2 : 1;
                }
                q++;
            }
            p = *q ? q + 2 : q;
        } else {
            p++;
        }
    }
    st->p = p;
}

static void bind_loop(struct state *st, const char *v1, int v1n,
                      const char *v2, int v2n, struct jinja_value base,
                      long n, long k) {
    struct jinja_value cur = seq_index(st, base, k);
    struct jinja_value lp = dict_make(st, 8);
    dict_put(st, lp, "index", jv_int(k + 1));
    dict_put(st, lp, "index0", jv_int(k));
    dict_put(st, lp, "first", jv_bool(k == 0));
    dict_put(st, lp, "last", jv_bool(k == n - 1));
    dict_put(st, lp, "length", jv_int(n));
    dict_put(st, lp, "previtem",
             k > 0 ? seq_index(st, base, k - 1) : jv_undef());
    dict_put(st, lp, "nextitem",
             k < n - 1 ? seq_index(st, base, k + 1) : jv_undef());
    if (v2) {
        int pair = cur.kind == JV_LIST &&
                   ((const struct list *)cur.node)->n >= 2;
        var_set(st, v1, v1n,
                pair ? ((const struct list *)cur.node)->items[0] : cur);
        var_set(st, v2, v2n,
                pair ? ((const struct list *)cur.node)->items[1]
                     : jv_undef());
    } else {
        var_set(st, v1, v1n, cur);
    }
    var_set(st, "loop", 4, lp);
}

static void do_for(struct state *st) {
    const char *v1 = arena_dup(st, st->ts, (size_t)st->tn);
    int v1n = (int)strlen(v1);
    const char *v2 = NULL;
    int v2n = 0;
    lex(st);
    if (is_op(st, ",")) {
        lex(st);
        v2 = arena_dup(st, st->ts, (size_t)st->tn);
        v2n = (int)strlen(v2);
        lex(st);
    }
    lex(st);
    struct jinja_value seq = eval_expr(st);
    consume_close(st);
    struct jinja_value *lp = var_find(st, "loop", 4);
    struct jinja_value loopsave = lp ? *lp : jv_undef();
    const char *body = st->p;
    int trim0 = st->trim_left;
    long n = seq_len(st, seq);
    int real = st->mute == 0 && n > 0;
    long iters = real ? n : 1;
    long k = 0;
    int stop = ST_ENDFOR;
    st->mute += real ? 0 : 1;
    while (k < iters) {
        st->p = body;
        st->trim_left = trim0;
        if (real) { bind_loop(st, v1, v1n, v2, v2n, seq, n, k); }
        stop = render_block(st);
        k++;
    }
    st->mute -= real ? 0 : 1;
    if (stop == ST_ELSE) {
        int show = st->mute == 0 && n == 0;
        consume_close(st);
        st->mute += show ? 0 : 1;
        render_block(st);
        st->mute -= show ? 0 : 1;
    }
    consume_close(st);
    var_set(st, "loop", 4, loopsave);
}

static void do_if(struct state *st) {
    int cond = truthy(st, eval_expr(st));
    consume_close(st);
    int taken_any = 0;
    int more = 1;
    while (more) {
        int take = cond && !taken_any;
        st->mute += !take;
        int stop = render_block(st);
        st->mute -= !take;
        taken_any = taken_any || take;
        if (stop == ST_ELIF) {
            cond = truthy(st, eval_expr(st));
            consume_close(st);
        } else if (stop == ST_ELSE) {
            cond = 1;
            consume_close(st);
        } else {
            if (stop == ST_ENDIF) { consume_close(st); }
            more = 0;
        }
    }
}

/* Walks template text: emits literals and {{...}}, skips {#...#}, recurses
 * into blocks, and returns the terminator that ended the enclosing block. */
static int render_block(struct state *st) {
    int stop = ST_NONE;
    if (++st->depth > MAX_DEPTH) {
        fail(st, "max recursion depth exceeded");
    }
    while (stop == ST_NONE) {
        const char *lit = st->p;
        while (*st->p && !(st->p[0] == '{' && (st->p[1] == '{' ||
               st->p[1] == '%' || st->p[1] == '#'))) {
            st->p++;
        }
        int two = st->p[0] == '{';
        int comment = two && st->p[1] == '#';
        int stmt = two && st->p[1] == '%';
        int lead = two && st->p[2] == '-';
        emit_lit(st, lit, st->p, lead);
        if (!*st->p) {
            stop = ST_EOF;
        } else if (comment) {
            while (*st->p && !(st->p[0] == '#' && st->p[1] == '}')) {
                st->p++;
            }
            st->trim_left = st->p[-1] == '-';
            st->p += *st->p ? 2 : 0;
        } else if (!stmt) {
            prime(st, st->p + 2 + lead);
            struct jinja_value v = eval_expr(st);
            const char *s = to_str(st, v);
            out_putn(st, s, strlen(s));
            consume_close(st);
        } else {
            prime(st, st->p + 2 + lead);
            if (is_id(st, "if")) {
                lex(st);
                do_if(st);
            } else if (is_id(st, "for")) {
                lex(st);
                do_for(st);
            } else if (is_id(st, "set")) {
                lex(st);
                do_set(st);
            } else if (is_id(st, "macro")) {
                lex(st);
                do_macro(st);
            } else if (is_id(st, "filter")) {
                lex(st);
                do_filter(st);
            } else if (is_id(st, "generation") ||
                       is_id(st, "endgeneration")) {
                lex(st);
                consume_close(st);
            } else if (is_id(st, "elif")) {
                lex(st);
                stop = ST_ELIF;
            } else if (is_id(st, "else")) {
                lex(st);
                stop = ST_ELSE;
            } else if (is_id(st, "endif")) {
                lex(st);
                stop = ST_ENDIF;
            } else if (is_id(st, "endfor")) {
                lex(st);
                stop = ST_ENDFOR;
            } else if (is_id(st, "endmacro")) {
                lex(st);
                stop = ST_ENDMACRO;
            } else if (is_id(st, "endset")) {
                lex(st);
                stop = ST_ENDSET;
            } else if (is_id(st, "endfilter")) {
                lex(st);
                stop = ST_ENDFILTER;
            } else {
                fprintf(stderr, "jinja: unsupported tag: %.*s\n",
                        st->tn, st->ts);
                fail(st, "tag");
            }
        }
    }
    st->depth--;
    return stop;
}

char *jinja_render(const char *tmpl, const struct jinja_host *host,
                   const struct jinja_var *vars, int n_vars) {
    struct state st = {0};
    char *result = NULL;
    size_t tlen = strlen(tmpl);
    st.host = host;
    st.cap_vars = n_vars + 16;
    st.vars = (struct jinja_var *)xalloc((size_t)st.cap_vars *
                                         sizeof(struct jinja_var));
    if (n_vars) {
        memcpy(st.vars, vars, (size_t)n_vars * sizeof(struct jinja_var));
    }
    st.n_vars = n_vars;
    /* Jinja strips one trailing template newline by default. */
    st.src = arena_dup(&st, tmpl, tlen && tmpl[tlen-1] == '\n'
                       ? tlen - 1 : tlen);
    st.p = st.src;
    if (setjmp(st.jmp) == 0) {
        render_block(&st);
        result = st.out.buf ? st.out.buf : (char *)oom(calloc(1, 1));
    } else {
        free(st.out.buf);
    }
    struct arena_blk *b = st.arena;
    while (b) {
        struct arena_blk *nx = b->next;
        free(b);
        b = nx;
    }
    free(st.vars);
    return result;
}

void jinja_free(char *s) {
    free(s);
}

/* ---------------------------------------------------------------------
 * SCOPE -- this engine targets the GGUF chat-template subset of Jinja2
 * (the llama.cpp/minja surface), not full 3.1.x. A single GGUF ships one
 * self-contained template rendered with messages/tools/flags, so the
 * cross-template machinery (inheritance, includes, imports) never fires
 * and is absent by design, not omission.
 *
 * Verified: byte-exact against the real Qwen3.5 GGUF template (9 cases),
 * and against real Jinja2 3.1.x golden output for the feature set below
 * (test/test_gen_jinja.py -> test/test_jinja.c).
 *
 * Implemented: literals (str/int/bool/none, list [..], dict {..}); and/or
 * (value semantics) / not / comparisons / in / is-tests / ~ concat;
 * arithmetic + - * / // % ** (integer; see division note below); ternary
 * incl. inline-if without else; a.b / a[i] / a['k'] / full
 * a[start:stop:step] slicing / filters / methods. Tests: string number
 * integer boolean none undefined defined true false mapping iterable
 * sequence even odd, comparison family equalto/eq/ne/lessthan(lt)/
 * greaterthan(gt)/le/ge via selectattr args (+ host). Filters: trim length
 * count tojson safe
 * string upper lower title capitalize int abs default(+falsy) first last
 * reverse sort(attribute=,reverse=) unique min max sum join map(+attribute=)
 * list items dictsort replace select reject selectattr rejectattr groupby.
 * Methods: startswith endswith split lstrip rstrip strip replace items get
 * keys (+ host). Statements: if/for(with loop.* and for-else)/set(=and
 * {% set %}block)/macro(positional+defaults+keyword args)/filter-block/
 * namespace/range/raise_exception/strftime_now/generation-passthrough/
 * comments. `x is defined`/`is undefined` recognise the builtin globals
 * (strftime_now, range, namespace, raise_exception) as defined, since the
 * engine resolves them by name rather than storing a variable.
 *
 * Deliberately excluded (fail fast via unsupported(), not silently
 * ignored): template inheritance/include/import; {% with %}, {% raw %},
 * {% autoescape %}, recursive for, line statements; trim_blocks/
 * lstrip_blocks and the {%+ %} override (this engine uses only explicit
 * {%- -%}); float values (ints only -- `/` is integer division and
 * `2**-1` is 1, where Jinja yields floats); **kwargs/varargs; the bare
 * `x is <test> y` argument grammar (the comparison family above is reached
 * through selectattr/rejectattr positional args, not `is equalto y`, and
 * `is divisibleby N` is unsupported); the long filter tail (batch, slice,
 * urlize, ...). Each is a localised add at its dispatch site.
 *
 * Loop scoping: `loop` is one variable, saved/restored across both macro
 * calls and nested for-loops, so an outer loop.* read after an inner loop
 * is correct. This is not general lexical scoping ({% set %} inside a for
 * does not follow Jinja's loop-local rules); fine for the targeted subset.
 * Unsupported constructs return NULL with a stderr breadcrumb -- see
 * unsupported(). minja is the ~2.5k-line reference for the full surface.
 * ------------------------------------------------------------------- */
