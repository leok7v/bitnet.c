#pragma once
#ifndef BN_GRAMMAR_H
#define BN_GRAMMAR_H

#include <stdbool.h>
#include <stdint.h>

/* A tiny regexp-style NFA for constraining LLM token sampling, per the
 * "regexp, not a CFG parser" design: programs are small fixed arrays of
 * byte-matching instructions, matching runs as a Thompson NFA over a
 * fixed-size active-state set (zero malloc). This is ~10% of GBNF's surface
 * but covers what tool-call / JSON masking actually needs: literal sequences,
 * character classes, "any byte until a sentinel" (free-text values), and
 * alternation / loops (the 0..N parameter repetition).
 *
 * The active-state set is the set of instruction indices the grammar could be
 * at right now. bn_grammar_accepts() asks "would committing this token string
 * keep at least one path alive?" -- the sampler masks every vocab token whose
 * answer is no. bn_grammar_advance() commits the chosen token.
 */

typedef enum {
    BN_G_CHAR,   /* consume one byte == a                                  */
    BN_G_RANGE,  /* consume one byte in [a, b]                             */
    BN_G_ANY,    /* consume any byte (except '\0')                         */
    BN_G_NOT,    /* consume any byte except a                             */
    BN_G_SPLIT,  /* epsilon: fork to next and alt                          */
    BN_G_JMP,    /* epsilon: go to next                                    */
    BN_G_MATCH,  /* accept: the grammar is satisfied here                  */
} BnGOp;

typedef struct {
    uint8_t op;     /* BnGOp                                               */
    uint8_t a, b;   /* CHAR/NOT: a; RANGE: a..b                            */
    int16_t next;   /* primary successor instruction index                */
    int16_t alt;    /* SPLIT secondary successor                          */
} BnGInst;

#define BN_G_MAX_PROG    256
#define BN_G_MAX_ACTIVE  128

typedef struct {
    const BnGInst *prog;
    int prog_len;
    int active[BN_G_MAX_ACTIVE];  /* char-consuming + MATCH instr indices */
    int n_active;
} BnGrammar;

/* Arm a grammar over prog; seeds the active set from the start instruction. */
void bn_grammar_init(BnGrammar *g, const BnGInst *prog, int prog_len);

/* Lookahead: would committing the whole byte string s keep a path alive?
 * Used to mask logits -- a token whose string is not acceptable is forbidden. */
bool bn_grammar_accepts(const BnGrammar *g, const char *s);

/* Commit an accepted token string, advancing the active set in place. */
void bn_grammar_advance(BnGrammar *g, const char *s);

/* MATCH reachable from the current active set -> the model MAY stop here. */
bool bn_grammar_can_stop(const BnGrammar *g);

/* No active states -> the grammar is dead (nothing can follow). */
bool bn_grammar_dead(const BnGrammar *g);

/* ---- fast logit masking over a whole vocabulary ------------------------- *
 * Naive masking would run bn_grammar_accepts() on all ~150k vocab strings per
 * generated token (O(vocab * token_len * states) -- halves decode). Instead we
 * sort the vocab once and walk it with a per-depth NFA-state stack: shared
 * prefixes are stepped once and a dead prefix prunes its whole run. Cost drops
 * to O(distinct trie edges * states) for the NFA work -- near-zero while the
 * grammar is tight, cheap while it is loose. No trie nodes are materialized. */

typedef struct BnGrammarVocab BnGrammarVocab;  /* opaque, malloc-backed */

/* Build the sorted view once from the tokenizer's vocab strings (borrowed,
 * must outlive the view). Returns NULL on OOM. */
BnGrammarVocab *bn_grammar_vocab_create(const char *const *strs, int n);
void            bn_grammar_vocab_destroy(BnGrammarVocab *v);

/* Set logits[t] = -INFINITY for every token g forbids from its current state. */
void bn_grammar_mask_logits(const BnGrammar *g, const BnGrammarVocab *v,
                            float *logits);

/* ---- builder: assemble a program without hand-counting jump targets ----- */

typedef struct {
    BnGInst inst[BN_G_MAX_PROG];
    int n;
} BnGBuilder;

void bn_gb_reset(BnGBuilder *b);
/* Emit one instruction; next < 0 means "fall through to the next index".
 * Returns the index written. */
int  bn_gb_emit(BnGBuilder *b, BnGOp op, uint8_t a, uint8_t bb,
                int next, int alt);
int  bn_gb_lit(BnGBuilder *b, const char *s);   /* literal byte run         */
/* Free run: consume any byte != c, zero or more, terminating at the first c
 * (which the next emitted instruction must match). Lets a value run until the
 * '<' of its closing tag, deterministically. */
int  bn_gb_until_char(BnGBuilder *b, uint8_t c);
/* Zero or more ASCII whitespace bytes (space, tab, newline, CR). Unlike
 * until_char, this does NOT swallow arbitrary non-delimiter text -- use it
 * between structural tokens so only real whitespace may sit there. */
int  bn_gb_ws_star(BnGBuilder *b);
int  bn_gb_match(BnGBuilder *b);                /* accept                   */
/* Patch a previously-emitted instruction's next/alt (for forward jumps and
 * loops). Pass -1 to leave a field unchanged. */
void bn_gb_patch(BnGBuilder *b, int idx, int next, int alt);

/* The canonical tool-call skeleton grammar, shared by the agent harness and the
 * one-shot repair path: forces the structural tags, leaves the function name and
 * each parameter name/value free. Returns the index of the MATCH instruction. */
int  bn_grammar_build_tool_call(BnGBuilder *b);

#endif /* BN_GRAMMAR_H */
