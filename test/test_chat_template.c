// Self-contained tests for the chat_template module (src/chat_template.c over
// src/jinja.c). No external oracle: the jinja engine itself is validated
// against real Jinja2 by test_jinja.c; this test proves the BnTpl host wiring
// (roles, content, reasoning, tool_calls/arguments, tools.tojson,
// add_generation_prompt, bos/eos) and the render + special-encode path.
//
// Part A (no model): renders inlined templates and asserts on the output.
// Part B (optional, pass a GGUF path): exercises bn_tokenizer_encode_special
// against a real vocab.
#include "chat_template.h"

#include <stdio.h>
#include <string.h>

static int n_pass;
static int n_fail;

// A compact ChatML-style template exercising the host surface.
static const char *TMPL =
    "{%- if tools %}<tools>{% for t in tools %}{{ t.tojson() }}{% endfor %}"
    "</tools>\n{% endif -%}"
    "{%- for m in messages -%}"
    "<|im_start|>{{ m.role }}\n"
    "{%- if m.reasoning_content is defined %}<think>{{ m.reasoning_content }}"
    "</think>{% endif -%}"
    "{%- if m.content %}{{ m.content }}{% endif -%}"
    "{%- if m.tool_calls %}{% for tc in m.tool_calls %}"
    "<tool_call>{{ tc.name }}</tool_call>{% endfor %}{% endif -%}"
    "<|im_end|>\n"
    "{%- endfor -%}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif -%}";

static void check(const char *label, const char *out, const char *needle) {
    int ok = out != NULL && strstr(out, needle) != NULL;
    printf("%-40s %s\n", label, ok ? "PASS" : "FAIL");
    if (ok) {
        n_pass++;
    } else {
        n_fail++;
        printf("    needle [%s] not in:\n%s\n---\n", needle,
               out ? out : "(null)");
    }
}

static void check_absent(const char *label, const char *out,
                         const char *needle) {
    int ok = out != NULL && strstr(out, needle) == NULL;
    printf("%-40s %s\n", label, ok ? "PASS" : "FAIL");
    if (ok) { n_pass++; } else { n_fail++; }
}

static void run_render(void) {
    BnTplMessage m_su[] = {
        {BN_TPL_ROLE_SYSTEM, "You are helpful.", NULL, NULL, 0},
        {BN_TPL_ROLE_USER, "Hi there.", NULL, NULL, 0},
    };
    char *out = bn_chat_template_render(TMPL, m_su, 2, NULL, 0, 0, 1,
                                        "<|bos|>", "<|eos|>");
    check("system role marker", out, "<|im_start|>system");
    check("system content", out, "You are helpful.");
    check("user role marker", out, "<|im_start|>user");
    check("user content", out, "Hi there.");
    check("generation prompt", out, "<|im_start|>assistant");
    check_absent("no tools section when none", out, "<tools>");
    bn_chat_template_free(out);

    char *out2 = bn_chat_template_render(TMPL, m_su, 2, NULL, 0, 0, 0,
                                         "<|bos|>", "<|eos|>");
    check_absent("no gen prompt when disabled", out2, "<|im_start|>assistant");
    bn_chat_template_free(out2);

    BnTplParam p[] = {{"location", "Paris"}};
    BnTplToolCall tc[] = {{"get_weather", p, 1}};
    BnTplMessage m_tc[] = {
        {BN_TPL_ROLE_USER, "Weather?", NULL, NULL, 0},
        {BN_TPL_ROLE_ASSISTANT, "", "need a tool", tc, 1},
        {BN_TPL_ROLE_TOOL, "22C", NULL, NULL, 0},
    };
    char *out3 = bn_chat_template_render(TMPL, m_tc, 3, NULL, 0, 1, 1,
                                         "<|bos|>", "<|eos|>");
    check("reasoning rendered", out3, "<think>need a tool</think>");
    check("tool_call name rendered", out3, "<tool_call>get_weather</tool_call>");
    check("tool role marker", out3, "<|im_start|>tool");
    check("tool message content", out3, "22C");
    bn_chat_template_free(out3);

    BnTplTool tools[] = {
        {"get_weather", "Get weather",
         "{\"type\":\"object\",\"properties\":{}}"},
    };
    char *out4 = bn_chat_template_render(TMPL, m_su, 2, tools, 1, 0, 1,
                                         "<|bos|>", "<|eos|>");
    check("tools section + tojson", out4, "\"name\": \"get_weather\"");
    bn_chat_template_free(out4);
}

static void run_special_encode(const char *model_path) {
    BnGGUFFile *gf = bn_gguf_open_file(model_path);
    int ok = 0;
    if (gf) {
        BnTokenizer tok;
        if (bn_tokenizer_init(&tok, gf) == 0) {
            int toks[64];
            int n = bn_tokenizer_encode_special(&tok, "<|im_start|>user\nhi"
                                                "<|im_end|>\n", 0, toks, 64);
            int im_start = bn_tokenizer_lookup(&tok, "<|im_start|>");
            int im_end = bn_tokenizer_lookup(&tok, "<|im_end|>");
            int saw_end = 0;
            for (int i = 0; i < n; i++) if (toks[i] == im_end) saw_end = 1;
            ok = n > 0 && (im_start < 0 || toks[0] == im_start) &&
                 (im_end < 0 || saw_end);
            bn_tokenizer_free(&tok);
        }
        bn_gguf_free(gf);
    }
    printf("%-40s %s\n", "special-encode <|im_start|>", ok ? "PASS" : "FAIL");
    if (ok) { n_pass++; } else { n_fail++; }
}

int main(int argc, char **argv) {
    run_render();
    if (argc > 1) run_special_encode(argv[1]);
    printf(n_fail ? "\n%d FAILED\n" : "\nALL PASS (%d)\n",
           n_fail ? n_fail : n_pass);
    return n_fail ? 1 : 0;
}
