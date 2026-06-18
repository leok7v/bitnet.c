#ifndef BN_TOKENIZER_H
#define BN_TOKENIZER_H

#include "gguf.h"

#define BN_BPE_UNICODE_OFFSET  0x100    // non-printable bytes → U+0100..U+0143
#define BN_BPE_UNICODE_END     0x143

typedef struct {
    char  **vocab;
    float  *scores;
    int     vocab_size;
    int     bos_id, eos_id, eot_id;
    int     im_start_id, im_end_id; // ChatML tokens (-1 if absent)
    int     endoftext_id;
    int     add_bos;                // whether to prepend BOS (from GGUF metadata)
    int     chatml;                 // 1 if model uses ChatML template
    int     metaspace;              // 1 for tokenizers that encode spaces as U+2581
    int     max_token_length;
    // internal: sorted index for binary search during encoding
    int    *sorted_indices;
    // internal: control/user-defined token ids, sorted by descending string
    // length, used to split special markers out before BPE (see encode_special)
    int    *special_ids;
    int     n_special;
} BnTokenizer;

int         bn_tokenizer_init(BnTokenizer *t, BnGGUFFile *f);
void        bn_tokenizer_free(BnTokenizer *t);
int         bn_tokenizer_encode(const BnTokenizer *t, const char *text, int add_bos,
                             int *tokens, int max_tokens);
// Like bn_tokenizer_encode, but recognizes special/control token literals
// (e.g. "<|im_start|>") embedded in the text and emits their single token id
// rather than byte-encoding them. Required for tokenizing rendered chat
// templates. Returns token count, or -1 on overflow/allocation failure.
int         bn_tokenizer_encode_special(const BnTokenizer *t, const char *text,
                             int add_bos, int *tokens, int max_tokens);
const char *bn_tokenizer_decode(const BnTokenizer *t, int token);
int         bn_tokenizer_lookup(const BnTokenizer *t, const char *str);

#endif // BN_TOKENIZER_H
