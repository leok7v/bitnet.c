#!/usr/bin/env python3
# Generates test/test_jinja.c from the cases below by rendering each with
# real Jinja2 (3.1.x) and baking the expected output in as C literals. The
# generated file is committed and self-contained: building/running the test
# needs only a C compiler, not Python or Jinja2. Regenerate after changing
# cases:  python3 test/test_gen_jinja.py  (requires: pip install jinja2)
#
# Jinja2 is the authority here. If the C engine disagrees with a golden
# value, the C is wrong (or the feature is genuinely unsupported and should
# fail fast) -- fix jinja.c, do not edit the expected output by hand.
import jinja2

# (name, template, context). Keep data shapes to dict/list/str/int/bool so
# the generated JSON-node host can model them, and pipe sequences through
# join/length rather than rendering raw (Python repr != engine repr). Use
# explicit - dashes for control flow so whitespace is deterministic and
# matches the engine (neither side uses trim_blocks/lstrip_blocks).
TESTS = [
    ("count", "{{ items|count }}", {"items": [1, 2, 3]}),
    ("reverse", "{{ items|reverse|join(',') }}", {"items": [1, 2, 3]}),
    ("sort", "{{ nums|sort|join(',') }}", {"nums": [3, 1, 2]}),
    ("sort_rev", "{{ nums|sort(reverse=true)|join(',') }}",
     {"nums": [3, 1, 2]}),
    ("sort_attr",
     "{{ people|sort(attribute='age')|map(attribute='name')|join(',') }}",
     {"people": [{"name": "a", "age": 30}, {"name": "b", "age": 20}]}),
    ("unique", "{{ xs|unique|join(',') }}", {"xs": [1, 1, 2, 3, 3]}),
    ("min_max", "{{ nums|min }}/{{ nums|max }}", {"nums": [3, 1, 2]}),
    ("sum", "{{ nums|sum }}", {"nums": [1, 2, 3]}),
    ("abs", "{{ neg|abs }}", {"neg": -5}),
    ("title", "{{ s|title }}", {"s": "hello world"}),
    ("select", "{{ flags|select|join(',') }}",
     {"flags": [0, 1, 2, 0, 3]}),
    ("reject", "{{ flags|reject|join(',') }}", {"flags": [0, 1, 2, 0, 3]}),
    ("selectattr",
     "{{ people|selectattr('active')|map(attribute='name')|join(',') }}",
     {"people": [{"name": "a", "active": True},
                 {"name": "b", "active": False}]}),
    ("rejectattr",
     "{{ people|rejectattr('active')|map(attribute='name')|join(',') }}",
     {"people": [{"name": "a", "active": True},
                 {"name": "b", "active": False}]}),
    ("map_upper", "{{ words|map('upper')|join(',') }}",
     {"words": ["a", "b"]}),
    ("map_attr", "{{ people|map(attribute='name')|join(',') }}",
     {"people": [{"name": "a"}, {"name": "b"}]}),
    ("groupby",
     "{% for k, g in items|groupby('cat') -%}"
     "{{ k }}:{{ g|map(attribute='name')|join('+') }};{%- endfor %}",
     {"items": [{"cat": "x", "name": "a"}, {"cat": "y", "name": "b"},
                {"cat": "x", "name": "c"}]}),
    ("dictsort",
     "{% for k, v in d|dictsort -%}{{ k }}={{ v }};{%- endfor %}",
     {"d": {"b": 2, "a": 1}}),
    ("default_falsy", "{{ ''|default('x', true) }}/{{ ''|default('x') }}",
     {}),
    ("replace", "{{ 'a.b'|replace('.', '_') }}", {}),
    ("capitalize", "{{ 'hi there'|capitalize }}", {}),
    ("trim", "{{ '  hi  '|trim }}", {}),
    ("even_odd",
     "{% for n in nums -%}{{ n }}:{{ 'E' if n is even else 'O' }};"
     "{%- endfor %}", {"nums": [1, 2, 3]}),
    ("is_number", "{{ x is number }}/{{ s is number }}",
     {"x": 3, "s": "a"}),
    ("is_string", "{{ s is string }}/{{ x is string }}",
     {"x": 3, "s": "a"}),
    ("is_mapping", "{{ d is mapping }}/{{ xs is mapping }}",
     {"d": {"a": 1}, "xs": [1]}),
    ("is_seq", "{{ xs is sequence }}/{{ x is sequence }}",
     {"xs": [1], "x": 3}),
    ("slice_from", "{{ xs[1:]|join(',') }}", {"xs": [1, 2, 3, 4]}),
    ("slice_to", "{{ xs[:2]|join(',') }}", {"xs": [1, 2, 3, 4]}),
    ("slice_mid", "{{ xs[1:3]|join(',') }}", {"xs": [1, 2, 3, 4]}),
    ("slice_rev", "{{ xs[::-1]|join(',') }}", {"xs": [1, 2, 3, 4]}),
    ("slice_step", "{{ xs[::2]|join(',') }}", {"xs": [1, 2, 3, 4]}),
    ("slice_negstart", "{{ xs[-2:]|join(',') }}", {"xs": [1, 2, 3, 4]}),
    ("slice_negstop", "{{ xs[:-1]|join(',') }}", {"xs": [1, 2, 3, 4]}),
    ("slice_full", "{{ xs[1:4:2]|join(',') }}", {"xs": [1, 2, 3, 4]}),
    ("slice_revstep", "{{ xs[3:0:-1]|join(',') }}", {"xs": [1, 2, 3, 4]}),
    ("dict_literal", "{{ {'a': 1, 'b': 2}|length }}", {}),
    ("dict_literal_sort",
     "{% for k, v in {'b': 2, 'a': 1}|dictsort -%}{{ k }}{{ v }}"
     "{%- endfor %}", {}),
    ("inline_if_true", "{{ 'yes' if cond }}", {"cond": True}),
    ("inline_if_false", "[{{ 'yes' if cond }}]", {"cond": False}),
    ("list_literal", "{{ [1, 2, 3]|join('-') }}", {}),
    ("or_value", "{{ a or 'def' }}", {"a": ""}),
    ("and_value", "{{ a and 'x' }}", {"a": "y"}),
    ("range1", "{{ range(3)|join(',') }}", {}),
    ("range2", "{{ range(2, 5)|join(',') }}", {}),
    ("range3", "{{ range(0, 10, 3)|join(',') }}", {}),
    ("set_block", "{%- set x %}hi{% endset -%}[{{ x }}]", {}),
    ("filter_block", "{%- filter upper -%}hi there{%- endfilter -%}", {}),
    ("for_else_empty",
     "{%- for i in xs -%}{{ i }}{%- else -%}none{%- endfor -%}",
     {"xs": []}),
    ("for_else_full",
     "{%- for i in xs -%}{{ i }}{%- else -%}none{%- endfor -%}",
     {"xs": [1, 2]}),
    ("nested_loop_scope",
     "{%- for a in outer -%}{%- for b in inner -%}{%- endfor -%}"
     "{{ loop.index }};{%- endfor -%}",
     {"outer": [1, 2], "inner": [9]}),
    ("kwargs",
     "{%- macro f(a, b=2) -%}{{ a }}-{{ b }}{%- endmacro -%}"
     "{{ f(1) }}|{{ f(1, b=9) }}", {}),
    ("mul", "{{ 6 * 7 }}", {}),
    ("mod", "{{ 7 % 3 }}", {}),
    ("mod_neg", "{{ -7 % 3 }}", {}),
    ("floordiv", "{{ 7 // 2 }}", {}),
    ("floordiv_neg", "{{ -7 // 2 }}", {}),
    ("pow", "{{ 2 ** 10 }}", {}),
    ("pow_unary", "{{ -2 ** 2 }}", {}),
    ("prec_muladd", "{{ 2 + 3 * 4 }}", {}),
    ("prec_parens", "{{ (2 + 3) * 4 }}", {}),
    ("mod_role",
     "{%- for n in nums -%}{{ n % 2 }}{%- endfor -%}",
     {"nums": [0, 1, 2, 3]}),
    ("mod_in_cmp",
     "{%- for i in nums -%}{{ 'A' if i % 2 == 0 else 'B' }}{%- endfor -%}",
     {"nums": [0, 1, 2, 3]}),
    ("dead_filter", "{% if False %}{{ x|nofilter }}{% endif %}ok", {}),
    ("dead_test", "{% if False %}{{ x is notest }}{% endif %}ok", {}),
    ("dead_function", "{% if False %}{{ nofunc() }}{% endif %}ok", {}),
    ("dead_strftime",
     "{% if False %}{{ strftime_now('%Y') }}{% endif %}ok", {}),
    ("eq_str_bool", "{{ 'hi' == false }}", {}),
    ("eq_emptystr_bool", "{{ '' == false }}", {}),
    ("eq_int_bool", "{{ 0 == false }}/{{ 1 == true }}/{{ 2 == true }}", {}),
    ("eq_str_int", "{{ '1' == 1 }}", {}),
    ("eq_none_bool", "{{ none == false }}/{{ none == none }}", {}),
    ("ne_str_bool_sentinel", "{{ 'hi' != false }}/{{ '' != false }}", {}),
    ("selectattr_equalto",
     "{{ people|selectattr('role', 'equalto', 'sys')|list|length }}",
     {"people": [{"role": "sys"}, {"role": "usr"}, {"role": "sys"}]}),
    ("rejectattr_equalto",
     "{{ people|rejectattr('role', 'equalto', 'sys')"
     "|map(attribute='role')|join(',') }}",
     {"people": [{"role": "sys"}, {"role": "usr"}]}),
    ("selectattr_gt",
     "{{ people|selectattr('age', 'greaterthan', 25)"
     "|map(attribute='name')|join(',') }}",
     {"people": [{"name": "a", "age": 30}, {"name": "b", "age": 20},
                 {"name": "c", "age": 40}]}),
    ("has_system_idiom",
     "{{ messages|selectattr('role', 'equalto', 'system')"
     "|list|length > 0 }}",
     {"messages": [{"role": "system"}, {"role": "user"}]}),
    ("has_system_absent",
     "{{ messages|selectattr('role', 'equalto', 'system')"
     "|list|length > 0 }}",
     {"messages": [{"role": "user"}]}),
    ("range_is_defined", "{{ range is defined }}", {}),
    ("namespace_is_defined", "{{ namespace is defined }}", {}),
    ("missing_is_defined", "{{ nope_xyz is defined }}", {}),
    ("missing_is_undefined", "{{ nope_xyz is undefined }}", {}),
    ("set_var_is_defined", "{% set x = 1 %}{{ x is defined }}", {}),
]


def cstr(s):
    out = []
    for ch in s:
        if ch == '\\':
            out.append('\\\\')
        elif ch == '"':
            out.append('\\"')
        elif ch == '\n':
            out.append('\\n')
        elif ch == '\t':
            out.append('\\t')
        elif ch == '\r':
            out.append('\\r')
        else:
            out.append(ch)
    return '"' + ''.join(out) + '"'


nodes = []
counter = [0]


def emit(v):
    i = counter[0]
    counter[0] += 1
    name = "jn%d" % i
    if isinstance(v, bool):
        nodes.append("static const JN %s={J_BOOL,0,0,%d,0,0,0,0};"
                      % (name, 1 if v else 0))
    elif isinstance(v, int):
        nodes.append("static const JN %s={J_NUM,0,%d,0,0,0,0,0};"
                      % (name, v))
    elif isinstance(v, str):
        nodes.append("static const JN %s={J_STR,%s,0,0,0,0,0,0};"
                      % (name, cstr(v)))
    elif isinstance(v, list):
        ids = [emit(e) for e in v]
        if ids:
            nodes.append("static const JN* %s_e[]={%s};"
                         % (name, ",".join("&" + x for x in ids)))
            ea = "%s_e" % name
        else:
            ea = "0"
        nodes.append("static const JN %s={J_ARR,0,0,0,0,0,%s,%d};"
                     % (name, ea, len(v)))
    elif isinstance(v, dict):
        ids = [emit(e) for e in v.values()]
        ks = ",".join(cstr(k) for k in v.keys())
        vs = ",".join("&" + x for x in ids)
        if v:
            nodes.append("static const char* %s_k[]={%s};" % (name, ks))
            nodes.append("static const JN* %s_v[]={%s};" % (name, vs))
            ka, va = "%s_k" % name, "%s_v" % name
        else:
            ka, va = "0", "0"
        nodes.append("static const JN %s={J_OBJ,0,0,0,%s,%s,0,%d};"
                     % (name, ka, va, len(v)))
    else:
        raise TypeError("unsupported test value: %r" % (v,))
    return name


def main():
    env = jinja2.Environment()
    tests = []
    for name, tmpl, ctx in TESTS:
        want = env.from_string(tmpl).render(**ctx)
        varids = [(k, emit(val)) for k, val in ctx.items()]
        tests.append((name, tmpl, want, varids))
    out = []
    out.append("/* GENERATED by test/test_gen_jinja.py -- do not edit.")
    out.append(" * Expected values come from real Jinja2; regenerate")
    out.append(" * rather than hand-editing. */")
    out.append('#include "jinja.h"')
    out.append("#include <stdio.h>")
    out.append("#include <stdlib.h>")
    out.append("#include <string.h>")
    out.append("")
    out.append("enum { J_OBJ=1, J_ARR, J_STR, J_NUM, J_BOOL };")
    out.append("typedef struct JN {")
    out.append("    int kind; const char *str; long num; int bln;")
    out.append("    const char **keys; const struct JN **vals;")
    out.append("    const struct JN **elems; int n;")
    out.append("} JN;")
    out.append("static struct jinja_value jvk(enum jinja_value_kind k){"
               "struct jinja_value v={0};v.kind=k;return v;}")
    out.append("static struct jinja_value wrap(const JN*j){"
               "struct jinja_value v={0};")
    out.append("    if(!j)return jvk(JV_UNDEFINED);")
    out.append("    if(j->kind==J_STR){v.kind=JV_STR;v.s=j->str;}")
    out.append("    else if(j->kind==J_NUM){v.kind=JV_INT;v.i=j->num;}")
    out.append("    else if(j->kind==J_BOOL){v.kind=JV_BOOL;v.i=j->bln;}")
    out.append("    else{v.kind=JV_NODE;v.tag=j->kind;v.node=j;"
               "v.i=j->n;}return v;}")
    out.append("static struct jinja_value g(void*c,struct jinja_value o,"
               "const char*n){(void)c;")
    out.append("    if(o.kind==JV_NODE&&o.tag==J_OBJ){const JN*j=o.node;")
    out.append("        for(int i=0;i<j->n;i++)"
               "if(!strcmp(j->keys[i],n))return wrap(j->vals[i]);}")
    out.append("    return jvk(JV_UNDEFINED);}")
    out.append("static struct jinja_value ix(void*c,struct jinja_value o,"
               "long i){(void)c;")
    out.append("    if(o.kind!=JV_NODE)return jvk(JV_UNDEFINED);")
    out.append("    const JN*j=o.node;")
    out.append("    if(j->kind==J_OBJ){struct jinja_value v={0};"
               "v.kind=JV_STR;v.s=j->keys[i];return v;}")
    out.append("    if(j->kind==J_ARR)return wrap(j->elems[i]);")
    out.append("    return jvk(JV_UNDEFINED);}")
    out.append("static long ln(void*c,struct jinja_value o){(void)c;"
               "return o.kind==JV_NODE?((const JN*)o.node)->n:0;}")
    out.append("static int tr(void*c,struct jinja_value v){(void)c;"
               "return v.kind==JV_NODE?((const JN*)v.node)->n!=0:0;}")
    out.append("static int ts(void*c,struct jinja_value v,const char*n){"
               "(void)c;")
    out.append("    if(v.kind!=JV_NODE)return 0;")
    out.append("    int k=((const JN*)v.node)->kind;")
    out.append("    if(!strcmp(n,\"mapping\"))return k==J_OBJ;")
    out.append("    if(!strcmp(n,\"sequence\")||!strcmp(n,\"iterable\"))"
               "return k==J_ARR;")
    out.append("    return 0;}")
    out.append("static struct jinja_value mt(void*c,struct jinja_value o,"
               "const char*n,struct jinja_value a){")
    out.append("    (void)c;(void)o;(void)n;(void)a;"
               "return jvk(JV_UNDEFINED);}")
    out.append("static struct jinja_host H={g,ix,ln,tr,ts,mt,0};")
    out.append("static int fails;")
    out.append("static void ck(const char*nm,const char*t,const char*w,"
               "struct jinja_var*v,int nv){")
    out.append("    char*o=jinja_render(t,&H,v,nv);")
    out.append("    int ok=o&&!strcmp(o,w);")
    out.append("    printf(\"%-20s %s\\n\",nm,ok?\"PASS\":\"FAIL\");")
    out.append("    if(!ok){fails++;printf(\"    want=[%s]\\n    got =[%s]"
               "\\n\",w,o?o:\"(null)\");}")
    out.append("    free(o);}")
    out.extend(nodes)
    for idx, (name, tmpl, want, varids) in enumerate(tests):
        out.append("static void t%d(void){" % idx)
        if varids:
            inits = ",".join('{%s,wrap(&%s)}' % (cstr(k), nid)
                             for k, nid in varids)
            out.append("    struct jinja_var v[]={%s};" % inits)
            out.append("    ck(%s,%s,%s,v,%d);"
                       % (cstr(name), cstr(tmpl), cstr(want), len(varids)))
        else:
            out.append("    ck(%s,%s,%s,0,0);"
                       % (cstr(name), cstr(tmpl), cstr(want)))
        out.append("}")
    out.append("int main(void){")
    for idx in range(len(tests)):
        out.append("    t%d();" % idx)
    out.append("    printf(fails?\"\\n%d FAILED\\n\":"
               "\"\\nALL GOLDEN TESTS PASS\\n\",fails);")
    out.append("    return fails?1:0;}")
    open("test/test_jinja.c", "w").write("\n".join(out) + "\n")
    print("wrote test/test_jinja.c with %d cases" % len(tests))


if __name__ == "__main__":
    main()
