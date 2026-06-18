#include "tools.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

static const char *find_arg(const ToolArg *args, int n_args,
                             const char *name) {
    for (int i = 0; i < n_args; i++) {
        if (args[i].name && strcmp(args[i].name, name) == 0)
            return args[i].value;
    }
    return NULL;
}

static char *rtrim(char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    while (n > 0) {
        unsigned char c = (unsigned char)s[n - 1];
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') s[--n] = '\0';
        else break;
    }
    return s;
}

static char *run_capture(const char *cmd) {
    FILE *p = popen(cmd, "r");
    if (!p) return NULL;
    size_t cap = 256, len = 0;
    char *buf = malloc(cap);
    if (!buf) { pclose(p); return NULL; }
    for (;;) {
        if (len + 256 >= cap) {
            cap *= 2;
            char *nb = realloc(buf, cap);
            if (!nb) { free(buf); pclose(p); return NULL; }
            buf = nb;
        }
        size_t r = fread(buf + len, 1, cap - len - 1, p);
        len += r;
        if (r == 0) break;
    }
    buf[len] = '\0';
    pclose(p);
    return buf;
}

static char *shell_escape(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    size_t cap = n * 4 + 8;
    char *out = malloc(cap);
    if (!out) return NULL;
    size_t i = 0;
    out[i++] = '\'';
    for (size_t k = 0; k < n; k++) {
        if (s[k] == '\'') {
            const char *esc = "'\\''";
            for (int j = 0; esc[j]; j++) out[i++] = esc[j];
        } else {
            out[i++] = s[k];
        }
    }
    out[i++] = '\'';
    out[i] = '\0';
    return out;
}

static char *json_string_field(const char *json, const char *key) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return NULL;
    p += strlen(needle);
    while (*p && *p != ':') p++;
    if (*p != ':') return NULL;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return NULL;
    p++;
    const char *start = p;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) p++;
        p++;
    }
    if (*p != '"') return NULL;
    size_t n = (size_t)(p - start);
    char *out = malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, start, n);
    out[n] = '\0';
    return out;
}

/* ---------- get_datetime ----------------------------------------- */

static const BnTplTool GET_DATETIME_TOOL = {
    "get_datetime",
    "Returns the current date and time.",
    "{\"type\":\"object\",\"properties\":{},\"required\":[]}"
};

const BnTplTool *tools_get_datetime_def(void) {
    return &GET_DATETIME_TOOL;
}

static char *run_get_datetime(const ToolArg *args, int n_args) {
    (void)args; (void)n_args;
    time_t now = time(NULL);
    struct tm local_tm;
    localtime_r(&now, &local_tm);
    char buf[64];
    strftime(buf, sizeof(buf), "%a %b %d %H:%M:%S %Y %Z", &local_tm);
    return strdup(buf);
}

/* ---------- read_file -------------------------------------------- */

static const BnTplTool READ_FILE_TOOL = {
    "read_file",
    "Read the contents of a file. Optionally specify a 1-based line range.",
    "{\"type\":\"object\","
    "\"properties\":{"
    "\"path\":{\"type\":\"string\"},"
    "\"start_line\":{\"type\":\"integer\"},"
    "\"end_line\":{\"type\":\"integer\"},"
    "\"append_loc\":{\"type\":\"boolean\","
    "\"description\":\"Prefix each line with its line number\"}"
    "},"
    "\"required\":[\"path\"]}"
};

const BnTplTool *tools_read_file_def(void) { return &READ_FILE_TOOL; }

static char *run_read_file(const ToolArg *args, int n_args) {
    const char *path = find_arg(args, n_args, "path");
    if (!path) return strdup("error: missing 'path' argument");
    const char *sl = find_arg(args, n_args, "start_line");
    const char *el = find_arg(args, n_args, "end_line");
    const char *al = find_arg(args, n_args, "append_loc");
    int start_line = sl ? atoi(sl) : 0;
    int end_line   = el ? atoi(el) : 0;
    int with_lnum  = al && (strcmp(al, "true") == 0 || strcmp(al, "1") == 0);
    FILE *f = fopen(path, "r");
    if (!f) {
        size_t cap = 64 + strlen(path);
        char *err = malloc(cap);
        if (err) snprintf(err, cap, "error: cannot open '%s'", path);
        return err;
    }
    size_t out_cap = 4096, out_len = 0;
    char *out = malloc(out_cap);
    if (!out) { fclose(f); return NULL; }
    out[0] = '\0';
    char line[4096];
    int lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        if (start_line > 0 && lineno < start_line) continue;
        if (end_line   > 0 && lineno > end_line)   break;
        char prefix[32];
        size_t plen = 0;
        if (with_lnum) {
            snprintf(prefix, sizeof(prefix), "%d: ", lineno);
            plen = strlen(prefix);
        }
        size_t llen = strlen(line);
        size_t need = out_len + plen + llen + 1;
        if (need > out_cap) {
            while (out_cap < need) out_cap *= 2;
            char *nb = realloc(out, out_cap);
            if (!nb) { free(out); fclose(f); return NULL; }
            out = nb;
        }
        if (with_lnum) { memcpy(out + out_len, prefix, plen); out_len += plen; }
        memcpy(out + out_len, line, llen);
        out_len += llen;
        out[out_len] = '\0';
    }
    fclose(f);
    return out;
}

/* ---------- file_glob_search ------------------------------------- */

static const BnTplTool FILE_GLOB_SEARCH_TOOL = {
    "file_glob_search",
    "Recursively search for files matching a glob pattern under a directory.",
    "{\"type\":\"object\","
    "\"properties\":{"
    "\"path\":{\"type\":\"string\",\"description\":\"Base directory\"},"
    "\"include\":{\"type\":\"string\","
    "\"description\":\"Glob pattern, e.g. **/*.cpp (default **)\"},"
    "\"exclude\":{\"type\":\"string\"}"
    "},"
    "\"required\":[\"path\"]}"
};

const BnTplTool *tools_file_glob_search_def(void) {
    return &FILE_GLOB_SEARCH_TOOL;
}

static char *run_file_glob_search(const ToolArg *args, int n_args) {
    const char *path    = find_arg(args, n_args, "path");
    const char *include = find_arg(args, n_args, "include");
    if (!path || !*path) path = ".";  // default to the working directory
    char *ep = shell_escape(path);
    if (!ep) return NULL;
    char *cmd;
    if (!include || !*include || strcmp(include, "**") == 0) {
        size_t cap = strlen(ep) + 32;
        cmd = malloc(cap);
        if (!cmd) { free(ep); return NULL; }
        snprintf(cmd, cap, "find %s -type f", ep);
    } else {
        const char *glob = include;
        if (strncmp(glob, "**/", 3) == 0) glob += 3;
        char *eg = shell_escape(glob);
        if (!eg) { free(ep); return NULL; }
        size_t cap = strlen(ep) + strlen(eg) + 32;
        cmd = malloc(cap);
        if (!cmd) { free(ep); free(eg); return NULL; }
        snprintf(cmd, cap, "find %s -type f -name %s", ep, eg);
        free(eg);
    }
    free(ep);
    char *out = run_capture(cmd);
    free(cmd);
    if (!out) return strdup("(no matches)");
    return rtrim(out);
}

/* ---------- grep_search ------------------------------------------ */

static const BnTplTool GREP_SEARCH_TOOL = {
    "grep_search",
    "Search for a regex pattern in files under a path. "
    "Returns matching lines.",
    "{\"type\":\"object\","
    "\"properties\":{"
    "\"path\":{\"type\":\"string\"},"
    "\"pattern\":{\"type\":\"string\"},"
    "\"include\":{\"type\":\"string\"},"
    "\"exclude\":{\"type\":\"string\"},"
    "\"return_line_numbers\":{\"type\":\"boolean\"}"
    "},"
    "\"required\":[\"path\",\"pattern\"]}"
};

const BnTplTool *tools_grep_search_def(void) { return &GREP_SEARCH_TOOL; }

static char *run_grep_search(const ToolArg *args, int n_args) {
    const char *path    = find_arg(args, n_args, "path");
    const char *pattern = find_arg(args, n_args, "pattern");
    const char *rln     = find_arg(args, n_args, "return_line_numbers");
    if (!path || !*path) path = ".";  // default to the working directory
    if (!pattern) return strdup("error: missing 'pattern' argument");
    int with_n = rln && (strcmp(rln, "true") == 0 || strcmp(rln, "1") == 0);
    char *ep = shell_escape(path);
    char *epat = shell_escape(pattern);
    if (!ep || !epat) { free(ep); free(epat); return NULL; }
    size_t cap = strlen(ep) + strlen(epat) + 32;
    char *cmd = malloc(cap);
    if (!cmd) { free(ep); free(epat); return NULL; }
    snprintf(cmd, cap, "grep -rE %s -- %s %s",
             with_n ? "-n" : "", epat, ep);
    free(ep); free(epat);
    char *out = run_capture(cmd);
    free(cmd);
    if (!out) return strdup("(no matches)");
    return rtrim(out);
}

/* ---------- exec_shell_command ----------------------------------- */

static const BnTplTool EXEC_SHELL_COMMAND_TOOL = {
    "exec_shell_command",
    "Execute a shell command and return its output "
    "(stdout and stderr combined).",
    "{\"type\":\"object\","
    "\"properties\":{"
    "\"command\":{\"type\":\"string\"},"
    "\"timeout\":{\"type\":\"integer\","
    "\"description\":\"Timeout in seconds (default 10, max 60)\"},"
    "\"max_output_size\":{\"type\":\"integer\","
    "\"description\":"
    "\"Maximum output size in bytes (default 16384)\"}"
    "},"
    "\"required\":[\"command\"]}"
};

const BnTplTool *tools_exec_shell_command_def(void) {
    return &EXEC_SHELL_COMMAND_TOOL;
}

static char *run_exec_shell_command(const ToolArg *args, int n_args) {
    const char *command  = find_arg(args, n_args, "command");
    const char *mos_str  = find_arg(args, n_args, "max_output_size");
    if (!command) return strdup("error: missing 'command' argument");
    int max_out = mos_str ? atoi(mos_str) : 16384;
    if (max_out <= 0) max_out = 16384;
    size_t cap = strlen(command) + 8;
    char *cmd = malloc(cap);
    if (!cmd) return NULL;
    snprintf(cmd, cap, "%s 2>&1", command);
    char *out = run_capture(cmd);
    free(cmd);
    if (!out) return strdup("");
    if ((int)strlen(out) > max_out) out[max_out] = '\0';
    return out;
}

/* ---------- write_file ------------------------------------------- */

static const BnTplTool WRITE_FILE_TOOL = {
    "write_file",
    "Write content to a file, creating it (including parent directories) "
    "if it does not exist. May use with edit_file for more complex edits.",
    "{\"type\":\"object\","
    "\"properties\":{"
    "\"path\":{\"type\":\"string\"},"
    "\"content\":{\"type\":\"string\"}"
    "},"
    "\"required\":[\"path\",\"content\"]}"
};

const BnTplTool *tools_write_file_def(void) { return &WRITE_FILE_TOOL; }

static char *run_write_file(const ToolArg *args, int n_args) {
    const char *path    = find_arg(args, n_args, "path");
    const char *content = find_arg(args, n_args, "content");
    if (!path)    return strdup("error: missing 'path' argument");
    if (!content) return strdup("error: missing 'content' argument");
    char *dir = strdup(path);
    if (!dir) return NULL;
    char *slash = strrchr(dir, '/');
    if (slash && slash != dir) {
        *slash = '\0';
        char *ep = shell_escape(dir);
        if (ep) {
            size_t cap = strlen(ep) + 16;
            char *cmd = malloc(cap);
            if (cmd) {
                snprintf(cmd, cap, "mkdir -p %s", ep);
                char *r = run_capture(cmd);
                free(r);
                free(cmd);
            }
            free(ep);
        }
    }
    free(dir);
    FILE *f = fopen(path, "w");
    if (!f) {
        size_t cap = 64 + strlen(path);
        char *err = malloc(cap);
        if (err) snprintf(err, cap, "error: cannot open '%s' for writing", path);
        return err;
    }
    size_t n = strlen(content);
    size_t written = fwrite(content, 1, n, f);
    fclose(f);
    size_t cap = 64 + strlen(path);
    char *out = malloc(cap);
    if (!out) return NULL;
    snprintf(out, cap, "wrote %zu bytes to %s", written, path);
    return out;
}

/* ---------- edit_file -------------------------------------------- */

static const BnTplTool EDIT_FILE_TOOL = {
    "edit_file",
    "Edit a file by applying a list of line-range changes.",
    "{\"type\":\"object\","
    "\"properties\":{"
    "\"path\":{\"type\":\"string\"},"
    "\"changes\":{\"type\":\"array\","
    "\"description\":\"List of changes to apply\","
    "\"items\":{"
    "\"type\":\"object\","
    "\"properties\":{"
    "\"mode\":{\"type\":\"string\","
    "\"enum\":[\"replace\",\"delete\",\"append\"]},"
    "\"line_start\":{\"type\":\"integer\","
    "\"description\":\"1-based; -1 for end of file\"},"
    "\"line_end\":{\"type\":\"integer\","
    "\"description\":\"Inclusive\"},"
    "\"content\":{\"type\":\"string\"}"
    "}}}"
    "},"
    "\"required\":[\"path\",\"changes\"]}"
};

const BnTplTool *tools_edit_file_def(void) { return &EDIT_FILE_TOOL; }

static int json_int_field(const char *json, const char *key, int *out) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return 0;
    p += strlen(needle);
    while (*p && *p != ':') p++;
    if (*p != ':') return 0;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '-' && !isdigit((unsigned char)*p)) return 0;
    *out = atoi(p);
    return 1;
}

/* Read all lines from file; return heap-allocated array of malloc'd strings.
   n_lines is set to the number of lines; returns NULL on error. */
static char **read_lines(const char *path, int *n_lines) {
    FILE *f = fopen(path, "r");
    if (!f) { *n_lines = 0; return NULL; }
    int cap = 64, n = 0;
    char **lines = malloc((size_t)cap * sizeof(char *));
    if (!lines) { fclose(f); *n_lines = 0; return NULL; }
    char buf[4096];
    while (fgets(buf, sizeof(buf), f)) {
        if (n >= cap) {
            cap *= 2;
            char **nb = realloc(lines, (size_t)cap * sizeof(char *));
            if (!nb) {
                for (int i = 0; i < n; i++) free(lines[i]);
                free(lines); fclose(f); *n_lines = 0; return NULL;
            }
            lines = nb;
        }
        lines[n++] = strdup(buf);
    }
    fclose(f);
    *n_lines = n;
    return lines;
}

static void free_lines(char **lines, int n) {
    for (int i = 0; i < n; i++) free(lines[i]);
    free(lines);
}

/* Write lines array to file. Returns 0 on success. */
static int write_lines(const char *path, char **lines, int n) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    for (int i = 0; i < n; i++) {
        if (lines[i]) fputs(lines[i], f);
    }
    fclose(f);
    return 0;
}

static char *run_edit_file(const ToolArg *args, int n_args) {
    const char *path    = find_arg(args, n_args, "path");
    const char *changes = find_arg(args, n_args, "changes");
    if (!path)    return strdup("error: missing 'path' argument");
    if (!changes) return strdup("error: missing 'changes' argument");
    char *mode_v    = json_string_field(changes, "mode");
    char *content_v = json_string_field(changes, "content");
    int line_start  = 1, line_end = 1;
    json_int_field(changes, "line_start", &line_start);
    json_int_field(changes, "line_end",   &line_end);
    if (!mode_v) {
        free(content_v);
        return strdup("error: could not parse changes; "
                      "use write_file or apply_diff");
    }
    int n_lines = 0;
    char **lines = read_lines(path, &n_lines);
    if (!lines && strcmp(mode_v, "append") != 0) {
        size_t cap = 64 + strlen(path);
        char *err = malloc(cap);
        free(mode_v); free(content_v);
        if (err) snprintf(err, cap, "error: cannot read '%s'", path);
        return err;
    }
    /* Clamp indices to valid range (1-based to 0-based). */
    int s0 = (line_start == -1) ? n_lines : line_start - 1;
    int e0 = (line_end   == -1) ? n_lines : line_end;
    if (s0 < 0) s0 = 0;
    if (e0 > n_lines) e0 = n_lines;
    /* Build output lines array. */
    int new_cap = n_lines + 64;
    char **out_lines = malloc((size_t)new_cap * sizeof(char *));
    if (!out_lines) {
        free_lines(lines, n_lines); free(mode_v); free(content_v);
        return NULL;
    }
    int oi = 0;
    for (int i = 0; i < s0 && i < n_lines; i++)
        out_lines[oi++] = strdup(lines[i]);
    if (strcmp(mode_v, "delete") != 0 && content_v) {
        /* Replace or append: insert content_v (ensure trailing newline). */
        size_t clen = strlen(content_v);
        int has_nl  = clen > 0 && content_v[clen - 1] == '\n';
        size_t need = clen + (has_nl ? 0 : 1) + 1;
        char *piece = malloc(need);
        if (piece) {
            memcpy(piece, content_v, clen);
            if (!has_nl) piece[clen++] = '\n';
            piece[clen] = '\0';
            out_lines[oi++] = piece;
        }
    }
    if (strcmp(mode_v, "append") != 0) {
        /* Skip replaced/deleted lines; copy the rest. */
        for (int i = e0; i < n_lines; i++)
            out_lines[oi++] = strdup(lines[i]);
    } else {
        /* Append: copy all original lines then the inserted piece. */
        /* (already copied up to s0; copy the rest from s0 onward) */
        for (int i = s0; i < n_lines; i++)
            out_lines[oi++] = strdup(lines[i]);
    }
    free_lines(lines, n_lines); free(mode_v); free(content_v);
    int rc = write_lines(path, out_lines, oi);
    free_lines(out_lines, oi);
    if (rc != 0) {
        size_t cap = 64 + strlen(path);
        char *err = malloc(cap);
        if (err) snprintf(err, cap, "error: cannot write '%s'", path);
        return err;
    }
    return strdup("ok");
}

/* ---------- apply_diff ------------------------------------------- */

static const BnTplTool APPLY_DIFF_TOOL = {
    "apply_diff",
    "Apply a unified diff to edit one or more files using git apply. "
    "Use this instead of edit_file when the changes are complex.",
    "{\"type\":\"object\","
    "\"properties\":{"
    "\"diff\":{\"type\":\"string\","
    "\"description\":"
    "\"Unified diff content in git diff format\"}"
    "},"
    "\"required\":[\"diff\"]}"
};

const BnTplTool *tools_apply_diff_def(void) { return &APPLY_DIFF_TOOL; }

static char *run_apply_diff(const ToolArg *args, int n_args) {
    const char *diff = find_arg(args, n_args, "diff");
    if (!diff) return strdup("error: missing 'diff' argument");
    const char *tmpfile = ".agent_apply.diff";
    FILE *f = fopen(tmpfile, "w");
    if (!f) return strdup("error: cannot write temp diff file");
    fputs(diff, f);
    fclose(f);
    char *cmd = malloc(strlen(tmpfile) + 32);
    if (!cmd) { remove(tmpfile); return NULL; }
    snprintf(cmd, strlen(tmpfile) + 32, "git apply %s 2>&1", tmpfile);
    char *out = run_capture(cmd);
    free(cmd);
    remove(tmpfile);
    if (!out || !*out) {
        free(out);
        return strdup("applied");
    }
    return rtrim(out);
}

/* ---------- dispatch --------------------------------------------- */

char *tools_execute(const char *function_name,
                    const ToolArg *args, int n_args) {
    if (!function_name) return NULL;
    if (strcmp(function_name, "get_datetime")        == 0)
        return run_get_datetime(args, n_args);
    if (strcmp(function_name, "read_file")           == 0)
        return run_read_file(args, n_args);
    if (strcmp(function_name, "file_glob_search")    == 0)
        return run_file_glob_search(args, n_args);
    if (strcmp(function_name, "grep_search")         == 0)
        return run_grep_search(args, n_args);
    if (strcmp(function_name, "exec_shell_command")  == 0)
        return run_exec_shell_command(args, n_args);
    if (strcmp(function_name, "write_file")          == 0)
        return run_write_file(args, n_args);
    if (strcmp(function_name, "edit_file")           == 0)
        return run_edit_file(args, n_args);
    if (strcmp(function_name, "apply_diff")          == 0)
        return run_apply_diff(args, n_args);
    size_t cap = 64 + strlen(function_name);
    char *out = malloc(cap);
    if (!out) return NULL;
    snprintf(out, cap, "error: unknown function '%s'", function_name);
    return out;
}

void tools_free_result(char *s) { free(s); }
