import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
import json
import websocket
import time
from typing import List, Dict, Any
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import csv
BASE_MODEL_PATH = "/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/finetuning/model/deepseek-coder-6.7b-instruct"
LORA_PATH = "/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/finetuning/train/deepseek_train20251217_30e/best_model"

# 检查CUDA是否可用
print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("当前设备:", device)

# 设置使用所有可用GPU
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"可用GPU数量: {device_count}")

# ==================== 阿里云百炼API工具函数 ====================
class BailianAPI:
    def __init__(self, api_key, model="deepseek-v3"):
        # deepseek-v3.2-exp
        self.api_key = api_key
        self.model = model
        # 初始化DashScope
        import dashscope
        dashscope.api_key = api_key

    def send_message(self, messages, max_retries=3):
        """发送消息到百炼大模型并获取回复"""
        from dashscope import Generation

        for attempt in range(max_retries):
            try:
                if isinstance(messages, list):
                    # 如果传入的是消息列表
                    prompt = "\n".join([msg.get("content", "") for msg in messages if msg.get("role") == "user"])
                else:
                    prompt = messages

                response = Generation.call(
                    model=self.model,
                    prompt=prompt,
                    temperature=0.1,
                    top_p=0.8,
                    result_format='message'
                )

                if response.status_code == 200:
                    return response.output.choices[0].message.content
                else:
                    print(f"第{attempt + 1}次尝试失败: {response.code} - {response.message}")
                    time.sleep(2)

            except Exception as e:
                print(f"第{attempt + 1}次尝试失败: {e}")
                time.sleep(2)

        raise Exception(f"所有{max_retries}次尝试都失败了")

    def extract_json_from_response(self, response):
        """从响应中提取JSON内容（保持原有逻辑）"""
        # 如果响应已经是字典或列表（已经解析过的JSON），直接返回
        if isinstance(response, (dict, list)):
            return response

            # 如果响应是字符串，尝试解析JSON
        if isinstance(response, str):
            # 先检查是否已经是JSON格式
            trimmed_response = response.strip()
            if (trimmed_response.startswith('[') and trimmed_response.endswith(']')) or \
                    (trimmed_response.startswith('{') and trimmed_response.endswith('}')):
                try:
                    return json.loads(trimmed_response)
                except json.JSONDecodeError:
                    pass

        # 尝试提取JSON部分
        json_pattern = r'```json\s*([\s\S]*?)\s*```|```\s*([\s\S]*?)\s*```|\[[\s\S]*\]|\{[\s\S]*\}'
        matches = re.findall(json_pattern, response)

        for match in matches:
            for group in match:
                if group.strip():
                    try:
                        return json.loads(group.strip())
                    except json.JSONDecodeError:
                        continue

        # 尝试直接解析整个响应
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 如果还是失败，尝试修复常见的JSON格式问题
        try:
            # 尝试修复单引号问题
            fixed_response = response.replace("'", '"')
            # 尝试修复无引号的key
            fixed_response = re.sub(r'(\w+):', r'"\1":', fixed_response)
            return json.loads(fixed_response)
        except json.JSONDecodeError:
            raise ValueError(f"无法从响应中提取JSON: {response}")

# 初始化百炼API
bailian_api = BailianAPI(
    api_key="sk-XXXXXXXXXXXXXXXXXX",  # 替换为你的API Key
    model="deepseek-v3"  # 可以选择其他模型如 "qwen-max", "qwen-plus"
)


# ==================== 1. 词法分析智能体 lexer ====================
class LexerAgent:
    def __init__(self):
        self.bailian_api = bailian_api

    def analyze(self, command: str) -> List[Dict[str, Any]]:
        """
        将Bash命令字符串分解为词法单元
        """
        safe_command = command.replace('\\', '\\\\').replace('"', '\\"').replace('\0', '\\\\0')
        prompt = f"""
        Act as a Bash command lexer. Please break down the following Bash command into lexical tokens. Return ONLY JSON format result, no other text.

        Input command: {safe_command}

        Requirements:
        1. Break down the command into minimal lexical units (tokens)
        2. Identify and classify various lexical elements
        3. Handle complex cases like quotes, escape characters, variable substitutions
        4. Return format MUST be: [{{"type": "type", "value": "value"}}]

        Token type definitions:
        - COMMAND: Command words (ls, grep, find, awk, etc.)
        - OPTION: Options (-l, --help, -rf, etc.)
        - ARGUMENT: Arguments (filenames, paths, string parameters)
        - OPERATOR: Operators (|, >, >>, <, &, &&, ||, ;, etc.)
        - SPECIAL: Special characters ((, ), {{, }}, $, etc.)
        - STRING: String content (inside quotes)
        - VARIABLE: Variables ($VAR, ${{VAR}}, etc.)
        - SUBCOMMAND: Subcommands ($(command), etc.)

        Example input: ls -l | grep "*.py" > out.txt
        Example output: [
            {{"type": "COMMAND", "value": "ls"}},
            {{"type": "OPTION", "value": "-l"}},
            {{"type": "OPERATOR", "value": "|"}},
            {{"type": "COMMAND", "value": "grep"}},
            {{"type": "STRING", "value": "*.py"}},
            {{"type": "OPERATOR", "value": ">"}},
            {{"type": "ARGUMENT", "value": "out.txt"}}
        ]
        """

        try:
            response = self.bailian_api.send_message([{"role": "user", "content": prompt}])

            # 使用改进的JSON提取方法
            tokens = self.bailian_api.extract_json_from_response(response)

            # 验证tokens格式
            if not isinstance(tokens, list):
                raise ValueError("返回的不是列表格式")

            for token in tokens:
                if not isinstance(token, dict) or 'type' not in token or 'value' not in token:
                    raise ValueError("token格式不正确")

            return tokens

        except Exception as e:
            print(f"词法分析错误: {e}")



# ==================== 2. 语法分析智能体 parser ====================
class ParserAgent:
    def __init__(self):
        self.bailian_api = bailian_api

    def analyze(self, tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        根据token列表构建抽象语法树(AST)
        """
        tokens_str = json.dumps(tokens, ensure_ascii=False)

        prompt = f"""
        Act as a Bash syntax parser. Please build an Abstract Syntax Tree (AST) based on the provided token list. Return ONLY JSON format result, no other text.

        Input tokens: {tokens_str}

        Requirements:
        1. Build AST according to Bash grammar rules
        2. Check for syntax errors and structural integrity
        3. Identify command structures, pipe chains, redirections, etc.
        4. Return format MUST be: {{"type": "node_type", "children": [child_nodes], "value": "value"}}

        Node type definitions:
        - COMMAND: Command node
        - PIPE: Pipe operation node
        - REDIRECT: Redirection node
        - LOGICAL: Logical operation node
        - SUBSHELL: Subshell node
        - ARGUMENT: Argument node
        - OPTION: Option node

        Example input: [
            {{"type": "COMMAND", "value": "ls"}},
            {{"type": "OPTION", "value": "-l"}},
            {{"type": "OPERATOR", "value": "|"}},
            {{"type": "COMMAND", "value": "grep"}},
            {{"type": "STRING", "value": "*.py"}}
        ]
        Example output: {{
            "type": "PIPE",
            "children": [
                {{
                    "type": "COMMAND",
                    "value": "ls",
                    "children": [
                        {{"type": "OPTION", "value": "-l"}}
                    ]
                }},
                {{
                    "type": "COMMAND",
                    "value": "grep",
                    "children": [
                        {{"type": "ARGUMENT", "value": "*.py"}}
                    ]
                }}
            ]
        }}
        """

        try:
            response = self.bailian_api.send_message([{"role": "user", "content": prompt}])

            ast = self.bailian_api.extract_json_from_response(response)
            return ast

        except Exception as e:
            print(f"语法分析错误: {e}")




# ==================== 3. 语义分析智能体 semantic ====================
class SemanticAgent:
    def __init__(self):
        self.bailian_api = bailian_api
        self.command_semantics = {
            "ls": "List directory contents",
            "grep": "Text search",
            "find": "Find files",
            "awk": "Text processing",
            "sed": "Stream editor",
            "cp": "Copy files",
            "mv": "Move files",
            "rm": "Delete files",
            "cat": "Concatenate and print files",
            "echo": "Display a line of text",
            "cd": "Change directory",
            "pwd": "Print working directory",
            "mkdir": "Create directory",
            "rmdir": "Remove directory",
            "chmod": "Change file permissions",
            "chown": "Change file owner",
            "ps": "Display process status",
            "top": "Display dynamic processes",
            "kill": "Terminate process",
            "tar": "Package and compress",
            "gzip": "Compress files",
            "gunzip": "Decompress files",
            "curl": "Transfer data",
            "wget": "Network download",
            "ping": "Test network connection"
        }

        self.option_semantics = {
            "-l": "Display in long format with detailed information",
            "-a": "Show all files (including hidden files)",
            "-r": "Recursive processing",
            "-f": "Force operation",
            "-h": "Human-readable format",
            "-v": "Show verbose information",
            "-i": "Interactive mode",
            "--help": "Display help information"
        }

    def analyze(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        对AST进行语义分析，添加语义标注
        """
        ast_str = json.dumps(ast, ensure_ascii=False)

        prompt = f"""
        Act as a Bash semantic analyzer. Please perform semantic analysis on the provided Abstract Syntax Tree (AST) and add semantic annotations. Return ONLY JSON format result, no other text.

        Input AST: {ast_str}

        Command semantics reference: {json.dumps(self.command_semantics, ensure_ascii=False)}
        Option semantics reference: {json.dumps(self.option_semantics, ensure_ascii=False)}

        Requirements:
        1. Parse the actual meaning of commands
        2. Analyze data flow (redirections, pipes, etc.)
        3. Add semantic annotations (command functionality descriptions, data flow directions, etc.)
        4. Return format: Add semantic field to the original AST structure

        Example input: {{
            "type": "PIPE",
            "children": [
                {{
                    "type": "COMMAND",
                    "value": "ls",
                    "children": [
                        {{"type": "OPTION", "value": "-l"}}
                    ]
                }},
                {{
                    "type": "COMMAND",
                    "value": "grep",
                    "children": [
                        {{"type": "ARGUMENT", "value": "*.py"}}
                    ]
                }}
            ]
        }}
        Example output: {{
            "type": "PIPE",
            "semantic": "Pipe operation, redirecting output of previous command as input to next command",
            "children": [
                {{
                    "type": "COMMAND",
                    "value": "ls",
                    "semantic": "List directory contents",
                    "children": [
                        {{
                            "type": "OPTION",
                            "value": "-l",
                            "semantic": "Use long listing format showing detailed information"
                        }}
                    ]
                }},
                {{
                    "type": "COMMAND",
                    "value": "grep",
                    "semantic": "Text search utility",
                    "children": [
                        {{
                            "type": "ARGUMENT",
                            "value": "*.py",
                            "semantic": "Search pattern matching all .py files"
                        }}
                    ]
                }}
            ]
        }}
        """

        try:
            response = self.bailian_api.send_message([{"role": "user", "content": prompt}])

            semantic_ast = self.bailian_api.extract_json_from_response(response)
            return semantic_ast

        except Exception as e:
            print(f"语义分析错误: {e}")


# ==================== 4. 生成解释 generator ====================
class GeneratorLLM:
    def __init__(self,
                 base_model_path: str = BASE_MODEL_PATH,
                 lora_path: str = LORA_PATH):
        """
        使用基座 Qwen3-8B + LoRA 适配器 的本地生成模型
        """
        self.model = None
        self.tokenizer = None
        self.device = None

        print(f"开始加载基座模型: {base_model_path}")
        # 很多 Qwen / Qwen3 模型需要 use_fast=False
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            use_fast=False
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        print(f"开始加载 LoRA 适配器: {lora_path}")
        self.model = PeftModel.from_pretrained(
            base_model,
            lora_path
        )

        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Qwen3-8B + LoRA 模型加载成功，设备: {self.device}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token



    def generate_explanation(self, semantic_ast:str) -> str:
        """
        根据语义增强的AST生成自然语言解释
        """

        # ast_str = json.dumps(semantic_ast, ensure_ascii=False)

        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a bash coding expert. Explain the following bash command in detail.<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"Command: {semantic_ast}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "Explanation:\n"
        )

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    length_penalty=1.0,
                    early_stopping=True,
                    max_new_tokens=128,

                    num_beams=4,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                )

            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # 如果里边保留了 "Explanation:" 之类前缀，做一次裁剪
            if "Explanation:" in explanation:
                explanation = explanation.split("Explanation:", 1)[-1].strip()

            return explanation

        except Exception as e:
            # 直接抛给外层，让 orchestrator 写到 error 字段里
            raise RuntimeError(f"解释生成错误: {e}")


# ==================== 5. 协调控制 orchestrator ====================
class Orchestrator:
    def __init__(self):
        self.lexer = LexerAgent()
        self.parser = ParserAgent()
        self.semantic = SemanticAgent()
        self.generator = GeneratorLLM()

    def process_command(self, command: str) -> Dict[str, Any]:
        """
        协调整个分析流程，处理Bash命令
        """
        result = {
            "original_command": command,
            "tokens": [],
            "ast": {},
            "semantic_ast": {},
            "explanation": "",
            "error": None
        }

        try:
            # 1. 词法分析
            tokens = self.lexer.analyze(command)
            result["tokens"] = tokens

            # 2. 语法分析
            ast = self.parser.analyze(tokens)
            result["ast"] = ast

            # 3. 语义分析
            semantic_ast = self.semantic.analyze(ast)
            result["semantic_ast"] = semantic_ast

            # 4. 生成解释
            explanation = self.generator.generate_explanation(command)
            result["explanation"] = explanation

        except Exception as e:
            result["error"] = str(e)
            print(f"处理命令时出错: {e}")

        return result

#     def format_output(self, result: Dict[str, Any]) -> str:
#         """
#         格式化输出结果
#         """
#         if result["error"]:
#             return f"错误: {result['error']}"
#
#         output = f"""
# 原始命令: {result['original_command']}
#
# 词法分析结果:
# {json.dumps(result['tokens'], indent=2, ensure_ascii=False)}
#
# 语法分析结果 (AST):
# {json.dumps(result['ast'], indent=2, ensure_ascii=False)}
#
# 语义分析结果:
# {json.dumps(result['semantic_ast'], indent=2, ensure_ascii=False)}
#
# 命令解释:
# {result['explanation']}
#         """
#
#         return output

def process_test_dataset(input_csv_path, output_txt_path, start_index=0):
    """
    批量处理测试数据集
    """
    # 初始化协调器
    orchestrator = Orchestrator()

    # 读取测试数据
    commands = []
    with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 2:  # 确保至少有命令和描述两列
                commands.append(row[0])  # 只取命令部分

    print(f"找到 {len(commands)} 条命令需要处理")

    # 处理每条命令并写入结果
    with open(output_txt_path, 'w', encoding='utf-8') as outfile:
        for i, command in enumerate(commands[start_index:], start=start_index + 1):
            print(f"处理第 {i}/{len(commands)} 条命令: {command}")

            try:
                result = orchestrator.process_command(command)

                if result["error"]:
                    explanation = f"错误: {result['error']}"
                else:
                    explanation = result["explanation"]

                # 写入格式: 命令 + 解释
                outfile.write(f"{explanation}\n")
                outfile.flush()  # 确保及时写入

                print(f"解释: {explanation[:150]}...")  # 显示前100个字符

            except Exception as e:
                error_msg = f"处理命令 '{command}' 时发生错误: {str(e)}"
                outfile.write(f"{error_msg}\n")
                print(error_msg)

    print(f"处理完成，结果已保存到: {output_txt_path}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 设置文件路径
    input_csv_path = "/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/dataset/test_simple.csv"
    output_txt_path = "/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/dataset/test_simple_hypotheses.txt"
    start_index = 0
    # 批量处理测试数据集
    process_test_dataset(input_csv_path, output_txt_path, start_index=start_index)

    # # 测试命令
    # test_command = 'find . -exec printf %s\0 {} ;'
    # # 创建协调器并处理命令
    # orchestrator = Orchestrator()
    # result = orchestrator.process_command(test_command)
    # # 输出简洁的结果
    # print(orchestrator.format_output(result))