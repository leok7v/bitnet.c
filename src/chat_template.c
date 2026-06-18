#include "chat_template.h"
#include "jinja.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Node tags dispatched through the jinja host callbacks. The engine treats
// jinja_value.node as opaque; these tags tell the callbacks which struct it is.
enum { T_MSGLIST = 1, T_MSG, T_TCLIST, T_TC, T_PARAMS, T_TOOLLIST, T_TOOL };

struct hostctx {
    char *json;       // scratch buffer for tojson, grown on demand
    size_t cap;
};

static struct jinja_value jv(enum jinja_value_kind k) {
    struct jinja_value v = {0};
    v.kind = k;
    return v;
}

static struct jinja_value vstr(const char *s) {
    struct jinja_value v = jv(JV_STR);
    v.s = s ? s : "";
    return v;
}

static struct jinja_value vbool(int b) {
    struct jinja_value v = jv(JV_BOOL);
    v.i = !!b;
    return v;
}

static struct jinja_value vnode(int tag, const void *p, long n) {
    struct jinja_value v = jv(JV_NODE);
    v.tag = tag;
    v.node = p;
    v.i = n;
    return v;
}

static const char *role_str(BnTplRole r) {
    const char *s = "user";
    if (r == BN_TPL_ROLE_SYSTEM) {
        s = "system";
    } else if (r == BN_TPL_ROLE_ASSISTANT) {
        s = "assistant";
    } else if (r == BN_TPL_ROLE_TOOL) {
        s = "tool";
    }
    return s;
}

// Serialize one tool definition as {"name":..,"description":..,"parameters":..}
// into the context scratch buffer; returned pointer is valid until the next call.
static const char *build_tool_json(struct hostctx *c, const BnTplTool *t) {
    const char *name = t->name ? t->name : "";
    const char *desc = t->description ? t->description : "";
    const char *params = t->params_json ? t->params_json : "{}";
    size_t need = strlen(name) + strlen(desc) * 2 + strlen(params) + 64;
    if (need > c->cap) {
        c->json = (char *)realloc(c->json, need);
        c->cap = need;
    }
    char *w = c->json;
    w += sprintf(w, "{\"name\": \"%s\", \"description\": \"", name);
    const char *p = desc;
    while (*p) {
        if (*p == '"' || *p == '\\') { *w++ = '\\'; }
        *w++ = *p++;
    }
    sprintf(w, "\", \"parameters\": %s}", params);
    return c->json;
}

static struct jinja_value host_get(void *ctx, struct jinja_value o,
                                   const char *nm) {
    (void)ctx;
    struct jinja_value r = jv(JV_UNDEFINED);
    if (o.tag == T_MSG) {
        const BnTplMessage *m = (const BnTplMessage *)o.node;
        if (!strcmp(nm, "role")) {
            r = vstr(role_str(m->role));
        } else if (!strcmp(nm, "content")) {
            r = m->content ? vstr(m->content) : jv(JV_NONE);
        } else if (!strcmp(nm, "reasoning_content")) {
            r = m->reasoning ? vstr(m->reasoning) : jv(JV_UNDEFINED);
        } else if (!strcmp(nm, "tool_calls")) {
            // Absent tool_calls reads as Undefined, not None: a real message dict
            // with no tool_calls key makes `message.tool_calls` Undefined and
            // `is defined` false, the way transformers templates expect. The
            // `message.get('tool_calls', none)` idiom still yields none, because
            // jinja's built-in get falls back to the caller default on a host miss.
            r = m->n_tool_calls > 0
                ? vnode(T_TCLIST, m->tool_calls, m->n_tool_calls)
                : jv(JV_UNDEFINED);
        }
    } else if (o.tag == T_TC) {
        const BnTplToolCall *tc = (const BnTplToolCall *)o.node;
        if (!strcmp(nm, "name")) {
            r = vstr(tc->function_name);
        } else if (!strcmp(nm, "arguments")) {
            r = vnode(T_PARAMS, tc, tc->n_params);
        }
    } else if (o.tag == T_PARAMS) {
        const BnTplToolCall *tc = (const BnTplToolCall *)o.node;
        int i = 0;
        while (i < tc->n_params) {
            if (!strcmp(tc->params[i].name, nm)) {
                r = vstr(tc->params[i].value);
            }
            i++;
        }
    }
    return r;
}

static struct jinja_value host_index(void *ctx, struct jinja_value o, long i) {
    (void)ctx;
    struct jinja_value r = jv(JV_UNDEFINED);
    if (o.tag == T_MSGLIST) {
        const BnTplMessage *m = (const BnTplMessage *)o.node;
        r = vnode(T_MSG, &m[i], 0);
    } else if (o.tag == T_TCLIST) {
        const BnTplToolCall *tc = (const BnTplToolCall *)o.node;
        r = vnode(T_TC, &tc[i], 0);
    } else if (o.tag == T_TOOLLIST) {
        const BnTplTool *t = (const BnTplTool *)o.node;
        r = vnode(T_TOOL, &t[i], 0);
    } else if (o.tag == T_PARAMS) {
        const BnTplToolCall *tc = (const BnTplToolCall *)o.node;
        r = vstr(tc->params[i].name);
    }
    return r;
}

static long host_len(void *ctx, struct jinja_value o) {
    (void)ctx;
    return o.i;
}

static int host_truthy(void *ctx, struct jinja_value v) {
    (void)ctx;
    int r = 1;
    if (v.tag == T_MSGLIST || v.tag == T_TCLIST || v.tag == T_TOOLLIST) {
        r = v.i != 0;
    }
    return r;
}

static int host_test(void *ctx, struct jinja_value v, const char *nm) {
    (void)ctx;
    int r = 0;
    if (!strcmp(nm, "mapping")) {
        r = v.tag == T_PARAMS;
    } else if (!strcmp(nm, "sequence")) {
        r = v.tag == T_MSGLIST || v.tag == T_TCLIST || v.tag == T_TOOLLIST;
    }
    return r;
}

static struct jinja_value host_method(void *ctx, struct jinja_value o,
                                      const char *nm, struct jinja_value arg) {
    (void)arg;
    struct jinja_value r = jv(JV_UNDEFINED);
    if (o.tag == T_TOOL && !strcmp(nm, "tojson")) {
        const BnTplTool *t = (const BnTplTool *)o.node;
        r = vstr(build_tool_json((struct hostctx *)ctx, t));
    }
    return r;
}

char *bn_chat_template_render(const char *tmpl,
                              const BnTplMessage *messages, int n_msgs,
                              const BnTplTool *tools, int n_tools,
                              int enable_thinking, int add_generation_prompt,
                              const char *bos_token, const char *eos_token) {
    char *out = NULL;
    if (tmpl) {
        struct hostctx hc = {0};
        struct jinja_host host = {host_get, host_index, host_len, host_truthy,
                                  host_test, host_method, &hc};
        struct jinja_var vars[6];
        int nv = 0;
        vars[nv].name = "messages";
        vars[nv++].value = vnode(T_MSGLIST, messages, n_msgs);
        // No tools -> None, not Undefined: matches apply_chat_template's tools=None
        // default. Templates branch on `{% if tools is none %}` and render
        // `{{ tools | tojson }}` in the else; Undefined would make `is none` false
        // and hand tojson an Undefined value, diverging from transformers.
        vars[nv].name = "tools";
        vars[nv++].value = n_tools > 0 ? vnode(T_TOOLLIST, tools, n_tools)
                                       : jv(JV_NONE);
        vars[nv].name = "enable_thinking";
        vars[nv++].value = vbool(enable_thinking);
        vars[nv].name = "add_generation_prompt";
        vars[nv++].value = vbool(add_generation_prompt);
        vars[nv].name = "bos_token";
        vars[nv++].value = bos_token ? vstr(bos_token) : jv(JV_UNDEFINED);
        vars[nv].name = "eos_token";
        vars[nv++].value = eos_token ? vstr(eos_token) : jv(JV_UNDEFINED);
        out = jinja_render(tmpl, &host, vars, nv);
        free(hc.json);
    }
    return out;
}

void bn_chat_template_free(char *s) {
    jinja_free(s);
}

int bn_chat_template_encode(const BnTokenizer *tok, const char *tmpl,
                            const BnTplMessage *messages, int n_msgs,
                            const BnTplTool *tools, int n_tools,
                            int enable_thinking, int add_generation_prompt,
                            const char *bos_token, const char *eos_token,
                            int *out_tokens, int max_tokens) {
    int n = -1;
    if (tok && out_tokens && max_tokens > 0) {
        char *rendered = bn_chat_template_render(tmpl, messages, n_msgs,
                                                 tools, n_tools, enable_thinking,
                                                 add_generation_prompt,
                                                 bos_token, eos_token);
        if (rendered) {
            n = bn_tokenizer_encode_special(tok, rendered, 0, out_tokens,
                                            max_tokens);
            bn_chat_template_free(rendered);
        }
    }
    return n;
}
