#ifndef JINJA_H
#define JINJA_H

#include <stddef.h>

/* Heterogeneous value. Strings are borrowed: they point either at host
 * memory, template literals, or the engine's transient arena. NODE is an
 * opaque host handle dispatched on `tag`; the engine never inspects it.
 * UNDEFINED is a missing variable/attribute (distinct from NONE); LIST is
 * an engine-owned sequence (e.g. the result of split), reached via `node`. */
enum jinja_value_kind {
    JV_NONE,
    JV_UNDEFINED,
    JV_BOOL,
    JV_INT,
    JV_STR,
    JV_NODE,
    JV_LIST,
};

struct jinja_value {
    enum jinja_value_kind kind;
    long i;
    const char *s;
    const void *node;
    int tag;
};

/* Everything the engine knows about your data lives behind these six
 * callbacks. The engine stays model-agnostic; per-model bytes live here. */
struct jinja_host {
    struct jinja_value (*get)(void *ctx, struct jinja_value obj,
                              const char *name);
    struct jinja_value (*index)(void *ctx, struct jinja_value obj, long i);
    long (*len)(void *ctx, struct jinja_value obj);
    int (*truthy)(void *ctx, struct jinja_value v);
    int (*test)(void *ctx, struct jinja_value v, const char *name);
    struct jinja_value (*method)(void *ctx, struct jinja_value obj,
                                 const char *name, struct jinja_value arg);
    void *ctx;
};

struct jinja_var {
    const char *name;
    struct jinja_value value;
};

/* Called on allocation failure; NULL by default (loader-zeroed), in which
 * case the library aborts. Assign a handler that longjmps or exits to
 * recover; if it returns, the library aborts anyway. Not safe to reassign
 * concurrently with rendering. */
extern void (*jinja_oom)(void);

/* Renders the template and returns a malloc'd string (free with
 * jinja_free). Returns NULL if a raise_exception fires or an unsupported
 * construct is hit; returns "" on empty output. `vars` seeds the global
 * scope (messages, tools, flags, bos_token, ...). */
char *jinja_render(const char *tmpl, const struct jinja_host *host,
                   const struct jinja_var *vars, int n_vars);

void jinja_free(char *s);

#endif
