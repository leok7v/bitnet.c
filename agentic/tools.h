#ifndef AGENTIC_TOOLS_H
#define AGENTIC_TOOLS_H

#include "chat_template.h"

typedef struct {
    const char *name;
    const char *value;
} ToolArg;

const BnTplTool *tools_get_datetime_def(void);
const BnTplTool *tools_read_file_def(void);
const BnTplTool *tools_file_glob_search_def(void);
const BnTplTool *tools_grep_search_def(void);
const BnTplTool *tools_exec_shell_command_def(void);
const BnTplTool *tools_write_file_def(void);
const BnTplTool *tools_edit_file_def(void);
const BnTplTool *tools_apply_diff_def(void);

char *tools_execute(const char *function_name,
                    const ToolArg *args, int n_args);

void tools_free_result(char *s);

#endif
