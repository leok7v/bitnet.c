#ifndef BN_CHAT_TEMPLATE_H
#define BN_CHAT_TEMPLATE_H

#include "tokenizer.h"

// Model-agnostic chat rendering: feed a GGUF tokenizer.chat_template (Jinja)
// plus a conversation, get back the formatted prompt or its tokens. The host
// data model mirrors what chat templates reference — roles, message content,
// reasoning, tool calls/arguments, and tool definitions.

typedef enum {
    BN_TPL_ROLE_SYSTEM,
    BN_TPL_ROLE_USER,
    BN_TPL_ROLE_ASSISTANT,
    BN_TPL_ROLE_TOOL,
} BnTplRole;

typedef struct {
    const char *name;
    const char *value;
} BnTplParam;

typedef struct {
    const char *function_name;
    const BnTplParam *params;
    int n_params;
} BnTplToolCall;

typedef struct {
    BnTplRole role;
    const char *content;
    const char *reasoning;            // reasoning_content (NULL if absent)
    const BnTplToolCall *tool_calls;
    int n_tool_calls;
} BnTplMessage;

typedef struct {
    const char *name;
    const char *description;
    const char *params_json;          // JSON schema string for "parameters"
} BnTplTool;

// Render `tmpl` with the given conversation. Returns a malloc'd string (free
// with bn_chat_template_free), or NULL on render failure / raise_exception.
char *bn_chat_template_render(const char *tmpl,
                              const BnTplMessage *messages, int n_msgs,
                              const BnTplTool *tools, int n_tools,
                              int enable_thinking, int add_generation_prompt,
                              const char *bos_token, const char *eos_token);

void  bn_chat_template_free(char *s);

// Render then tokenize with special-token splitting (bn_tokenizer_encode_special).
// Returns token count, or -1 on render/tokenize failure.
int   bn_chat_template_encode(const BnTokenizer *tok, const char *tmpl,
                              const BnTplMessage *messages, int n_msgs,
                              const BnTplTool *tools, int n_tools,
                              int enable_thinking, int add_generation_prompt,
                              const char *bos_token, const char *eos_token,
                              int *out_tokens, int max_tokens);

#endif // BN_CHAT_TEMPLATE_H
